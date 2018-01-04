/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples.mllib

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.feature.PPCA
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.SparkSession


object PPCATest {

  case class parsedCSV(df: DataFrame, m: Integer, n: Integer, meanVec: DenseVector)

  def fromCSV(input: String, sparkSession: SparkSession): parsedCSV = {
    // Load and parse the data
    val customSchema = StructType(Array(
      StructField("user", StringType, true),
      StructField("item", StringType, true),
      StructField("rating", DoubleType, true),
      StructField("timestamp", StringType, true)
    ))

    val data = sparkSession.read.format("com.databricks.spark.csv").schema(customSchema).csv(input)
    val selectedData = data.select("user", "item", "rating")

    val userindexer = new StringIndexer()
      .setInputCol("user")
      .setOutputCol("userNumber")
    val itemindexer = new StringIndexer()
      .setInputCol("item")
      .setOutputCol("itemNumber")

    val modelu = userindexer.fit(selectedData)

    val m = modelu.labels.length

    val df1 = modelu.transform(selectedData)

    val modeli = itemindexer.fit(df1)

    val n = modeli.labels.length

    val df2 = modeli.transform(df1)

    val df3 = df2.select("userNumber", "itemNumber", "rating")

    //    df3.select("itemNumber", "rating").groupBy("itemNumber").mean("rating").show

    val mean = df3.select("itemNumber", "rating").groupBy(
      "itemNumber").mean("rating").select("avg(rating)").rdd.map{
      row => row(0).asInstanceOf[Double]}.collect()

    val meanVector = new DenseVector(mean)

    // print(meanVector)

    parsedCSV(df3, m, n, meanVector)


  }

  def fromCSV(filepath: String) : DenseMatrix = {
    val bufferedSource = Source.fromFile(filepath)
    var ncols: Int = 0
    val values = ArrayBuffer[Double]()
    for (line <- bufferedSource.getLines) {
      ncols +=1
      val cols = line.split(" ").map(_.trim).map(s => s.toDouble)
      values ++= cols
    }
    bufferedSource.close
    val nrows: Int = values.size/ncols
    new DenseMatrix(nrows, ncols, values.toArray)

  }

  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.
      builder.appName("PPCA").getOrCreate()

    val sc = sparkSession.sparkContext

    val ratings = fromCSV(args(3).toString, sparkSession)

    val nrow = ratings.m

    val ncol = ratings.n

    val ym = ratings.meanVec

    val entrys = ratings.df.rdd.map(row => MatrixEntry(row(0).asInstanceOf[Double].toLong,
      row(1).asInstanceOf[Double].toLong, row(2).asInstanceOf[Double]))
    val mat = new CoordinateMatrix(entrys, nrow.toLong, ncol.toLong)

    val ratingsIdx = mat.toIndexedRowMatrix()
    val ratingsRow = ratingsIdx.toRowMatrix()


    val ppca = PPCA(args(0).toInt, args(1).toDouble, args(2).toDouble)

    ppca.computePCA(ratingsRow, ym)

    sc.stop()
  }

}
