from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, asc, month, year, dayofmonth, monotonically_increasing_id
from pyspark.sql.types import IntegerType, DoubleType, DateType

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("CrimeProject") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to reduce console noise
    spark.sparkContext.setLogLevel("ERROR")

    print("Reading CSV data...")
    
    # Load the source file
    raw_df = spark.read \
        .option("header", "true") \
        .option("delimiter", ";") \
        .option("inferSchema", "true") \
        .csv("CrimeData.csv")

    # Data Type Casting
    # Explicitly cast columns to ensure correct data types (inferSchema is not always accurate)
    df = raw_df.withColumn("DateOccured", col("DateOccured").cast(DateType())) \
               .withColumn("AreaCode", col("AreaCode").cast(IntegerType())) \
               .withColumn("CrimeCode", col("CrimeCode").cast(IntegerType())) \
               .withColumn("PremisCode", col("PremisCode").cast(IntegerType())) \
               .withColumn("WeaponCode", col("WeaponCode").cast(IntegerType())) \
               .withColumn("VictimAge", col("VictimAge").cast(IntegerType())) \
               .withColumn("Latitude", col("Latitude").cast(DoubleType())) \
               .withColumn("Longitude", col("Longitude").cast(DoubleType()))

    # --- Star Schema Creation ---
    print("Creating Dimensions and Fact table...")

    # 1. Area Dimension
    areas = df.select("AreaCode", "Area").distinct()
    areas = areas.withColumnRenamed("Area", "AreaName")

    # 2. Crime Dimension
    crimes = df.select("CrimeCode", "CrimeDescription").distinct()

    # 3. Premis Dimension
    premis = df.select("PremisCode", "PremisDescription").distinct()

    # 4. Weapon Dimension
    weapons = df.select("WeaponCode", "Weapon").distinct()

    # 5. Status Dimension
    statuses = df.select("CaseStatusCode", "CaseStatusDescription").distinct()

    # 6. Victim Dimension
    # Since a unique Victim ID doesn't exist, we generate one using monotonically_increasing_id
    victims = df.select("VictimSex", "VictimDescentCode", "VictimDescent").distinct()
    victims = victims.withColumn("VictimID", monotonically_increasing_id())

    # 7. Date Dimension
    dates = df.select("DateOccured").distinct()
    dates = dates.withColumn("Year", year("DateOccured"))
    dates = dates.withColumn("Month", month("DateOccured"))
    dates = dates.withColumn("Day", dayofmonth("DateOccured"))

    # Create the Fact Table (FactCrime)
    # Join with the victims dimension to retrieve the generated VictimID
    fact_df = df.join(victims, 
        (df.VictimSex == victims.VictimSex) & 
        (df.VictimDescentCode == victims.VictimDescentCode) &
        (df.VictimDescent == victims.VictimDescent), "left")
    
    # Select only the foreign keys and metrics for the Fact table
    fact_crime = fact_df.select(
        "CaseID", "DateOccured", "AreaCode", "CrimeCode", 
        "PremisCode", "WeaponCode", "CaseStatusCode", 
        "VictimID", "VictimAge", "Latitude", "Longitude"
    )

    # Cache the Fact table to optimize query performance
    fact_crime.cache()

    # --- Queries / Reports ---
    print("Generating Reports...")

    # Report 1: Incidents per Area & Premise
    # Join Fact table with Area and Premise dimensions
    r1_join = fact_crime.join(areas, "AreaCode").join(premis, "PremisCode")
    r1 = r1_join.groupBy("AreaName", "PremisDescription").count()
    # Sort results
    r1 = r1.orderBy(asc("AreaName"), desc("count")).withColumnRenamed("count", "Total")

    # Report 2: Top 10 Crimes
    r2 = fact_crime.join(crimes, "CrimeCode") \
        .groupBy("CrimeDescription").count() \
        .orderBy(desc("count")) \
        .limit(10)
    r2 = r2.withColumnRenamed("count", "Total")

    # Report 3: Monthly Statistics
    # Requires joining with the Date dimension
    r3 = fact_crime.join(dates, "DateOccured") \
        .groupBy("Year", "Month").count() \
        .orderBy(asc("Year"), asc("Month"))
    r3 = r3.withColumnRenamed("count", "Total")

    # Report 4: Case Status per Crime Type
    r4 = fact_crime.join(crimes, "CrimeCode").join(statuses, "CaseStatusCode") \
        .groupBy("CrimeDescription", "CaseStatusDescription").count() \
        .orderBy("CrimeDescription", "CaseStatusDescription")
    r4 = r4.withColumnRenamed("count", "Total")

    # Report 5: Data Cube (Victim Statistics)
    # Perform multi-dimensional analysis using the cube function
    r5 = fact_crime.join(victims, "VictimID") \
        .cube("VictimDescent", "VictimSex", "VictimAge").count() \
        .orderBy("VictimDescent", "VictimSex", "VictimAge")
    r5 = r5.withColumnRenamed("count", "Total")

    # --- Export to CSV ---
    print("Saving results to CSV...")

    # Save reports (using coalesce(1) to output a single file per report)
    r1.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report1_AreaPremis")
    r2.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report2_TopCrimes")
    r3.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report3_Monthly")
    r4.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report4_Status")
    r5.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report5_Cube")

    # Save Star Schema Tables
    areas.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Area")
    crimes.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Crime")
    premis.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Premis")
    weapons.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Weapon")
    statuses.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Status")
    victims.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Victim")
    dates.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Date")
    fact_crime.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Fact_Crime")

    print("Program completed successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
