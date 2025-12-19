from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, asc, month, year, dayofmonth, monotonically_increasing_id
from pyspark.sql.types import IntegerType, DoubleType, DateType

def main():
    # Arxikopoiisi Spark
    spark = SparkSession.builder \
        .appName("CrimeProject") \
        .master("local[*]") \
        .getOrCreate()

    # Gia na min bgazei polla logs
    spark.sparkContext.setLogLevel("ERROR")

    print("Reading CSV data...")
    
    # Fortosi tou arxeiou
    raw_df = spark.read \
        .option("header", "true") \
        .option("delimiter", ";") \
        .option("inferSchema", "true") \
        .csv("CrimeData.csv")

    # Metatropi ton typon dedomenon (Casting)
    # Metatrepoume ta pedia se sosta types giati to inferschema den ta pianei panta sosta
    df = raw_df.withColumn("DateOccured", col("DateOccured").cast(DateType())) \
               .withColumn("AreaCode", col("AreaCode").cast(IntegerType())) \
               .withColumn("CrimeCode", col("CrimeCode").cast(IntegerType())) \
               .withColumn("PremisCode", col("PremisCode").cast(IntegerType())) \
               .withColumn("WeaponCode", col("WeaponCode").cast(IntegerType())) \
               .withColumn("VictimAge", col("VictimAge").cast(IntegerType())) \
               .withColumn("Latitude", col("Latitude").cast(DoubleType())) \
               .withColumn("Longitude", col("Longitude").cast(DoubleType()))

    # --- Dimiourgia Star Schema ---
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
    # Epeidi den exoume victim ID, ftiaxnoume ena diko mas me to monotonically_increasing_id
    victims = df.select("VictimSex", "VictimDescentCode", "VictimDescent").distinct()
    victims = victims.withColumn("VictimID", monotonically_increasing_id())

    # 7. Date Dimension
    dates = df.select("DateOccured").distinct()
    dates = dates.withColumn("Year", year("DateOccured"))
    dates = dates.withColumn("Month", month("DateOccured"))
    dates = dates.withColumn("Day", dayofmonth("DateOccured"))

    # Dimiourgia tou Fact Table (FactCrime)
    # Kanoume join me to victim table gia na paroume to VictimID
    fact_df = df.join(victims, 
        (df.VictimSex == victims.VictimSex) & 
        (df.VictimDescentCode == victims.VictimDescentCode) &
        (df.VictimDescent == victims.VictimDescent), "left")
    
    # Kratame mono ta kleidia kai ta metrics
    fact_crime = fact_df.select(
        "CaseID", "DateOccured", "AreaCode", "CrimeCode", 
        "PremisCode", "WeaponCode", "CaseStatusCode", 
        "VictimID", "VictimAge", "Latitude", "Longitude"
    )

    # Cache gia kalyteri taxytita sta queries
    fact_crime.cache()

    # --- Queries / Reports ---
    print("Generating Reports...")

    # Report 1: Incidents per Area & Premis
    # Join fact me area kai premis
    r1_join = fact_crime.join(areas, "AreaCode").join(premis, "PremisCode")
    r1 = r1_join.groupBy("AreaName", "PremisDescription").count()
    # Taksinomisi
    r1 = r1.orderBy(asc("AreaName"), desc("count")).withColumnRenamed("count", "Total")

    # Report 2: Top 10 Crimes
    r2 = fact_crime.join(crimes, "CrimeCode") \
        .groupBy("CrimeDescription").count() \
        .orderBy(desc("count")) \
        .limit(10)
    r2 = r2.withColumnRenamed("count", "Total")

    # Report 3: Monthly stats
    # Xreiazomaste to Date dimension
    r3 = fact_crime.join(dates, "DateOccured") \
        .groupBy("Year", "Month").count() \
        .orderBy(asc("Year"), asc("Month"))
    r3 = r3.withColumnRenamed("count", "Total")

    # Report 4: Status per Crime
    r4 = fact_crime.join(crimes, "CrimeCode").join(statuses, "CaseStatusCode") \
        .groupBy("CrimeDescription", "CaseStatusDescription").count() \
        .orderBy("CrimeDescription", "CaseStatusDescription")
    r4 = r4.withColumnRenamed("count", "Total")

    # Report 5: Data Cube (Victim stats)
    # Erotima me cube
    r5 = fact_crime.join(victims, "VictimID") \
        .cube("VictimDescent", "VictimSex", "VictimAge").count() \
        .orderBy("VictimDescent", "VictimSex", "VictimAge")
    r5 = r5.withColumnRenamed("count", "Total")

    # --- Eksagogi se CSV ---
    print("Saving results to CSV...")

    # Sosimo ton apotelesmaton (xrisimopoio coalesce(1) gia na bgei ena arxeio)
    
    # Reports output
    r1.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report1_AreaPremis")
    r2.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report2_TopCrimes")
    r3.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report3_Monthly")
    r4.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report4_Status")
    r5.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_Reports/Report5_Cube")

    # Schema output
    areas.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Area")
    crimes.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Crime")
    premis.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Premis")
    weapons.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Weapon")
    statuses.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Status")
    victims.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Victim")
    dates.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Dim_Date")
    fact_crime.coalesce(1).write.mode("overwrite").option("header", "true").csv("Outputs_StarSchema/Fact_Crime")

    print("end of the programm.")
    spark.stop()

if __name__ == "__main__":
    main()
