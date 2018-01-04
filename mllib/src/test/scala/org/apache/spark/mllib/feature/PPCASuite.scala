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

package org.apache.spark.mllib.feature

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random

import breeze.linalg.inv

import org.apache.spark.SparkFunSuite
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{BLAS, DenseMatrix, DenseVector, Matrices, Matrix, SparseMatrix,
SparseVector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, RowMatrix}
import org.apache.spark.mllib.util.MLlibTestSparkContext

case class YtXXtX(centralYtX: Matrix, centralXtX: Matrix)

class PPCASuite extends SparkFunSuite with Logging with MLlibTestSparkContext{
  val v1 = new SparseVector(3, Array(0, 1, 2), Array(1.0, 4.0, 7.0))
  val v2 = new SparseVector(3, Array(0, 1, 2), Array(2.0, 5.0, 8.0))
  val v3 = new SparseVector(3, Array(0, 1, 2), Array(3.0, 6.0, 9.0))
  val vectors = Seq(v1, v2, v3)

  val nRows = 3L
  val nCols = 3L

  val d = 3

  val mean = new DenseVector(Array(2.0, 5.0, 8.0))
  val norm = 6.0

  var C = new DenseMatrix(nCols.toInt, d,
    Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
  var ss = 3.0

  var YtX = Matrices.dense(nCols.toInt, d,
    Array(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

  var XtX = Matrices.dense(nCols.toInt, d,
    Array(0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125))


  var CtC = Matrices.fromBreeze(C.transpose.asBreeze * C.asBreeze)

  var M = Matrices.fromBreeze(CtC.asBreeze
    + ss * DenseMatrix.eye(d).asBreeze)

  M = Matrices.fromBreeze(inv(M.asBreeze.toDenseMatrix))

  //  M = M.transpose

  //  val CM = Matrices.fromBreeze((C.asBreeze * M.asBreeze).toDenseMatrix)
  val CM = new DenseMatrix(nCols.toInt, d,
    Array(0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25))
  val xm = Vectors.fromBreeze(CM.transpose.asBreeze
    * mean.asBreeze).toDense

  var rowMat: RowMatrix = _
  var largeMat: RowMatrix = _
  var ratings: RowMatrix = _

  var users: Int = 0
  var items: Int = 0

  override def beforeAll() {
    super.beforeAll()

    rowMat = new RowMatrix(sc.parallelize(vectors))

    val m = 40
    val n = 20
    val rows = sc.parallelize(0 until m, 2).mapPartitionsWithIndex { (idx, iter) =>
      val random = new Random(idx)
      iter.map(i => Vectors.dense(Array.fill(n)(random.nextDouble())))
    }

    //    largeMat = new RowMatrix(rows)
    //
    //    val pcainput = new PCAInput()
    //    ratings = pcainput.fromCSV(
    // "/home/han/git/dataset/ratings_Amazon_Instant_Video.csv", spark.sqlContext).toRowMatrix()
    //    users = ratings.numRows().toInt
    //    items = ratings.numCols().toInt

  }

  test("meanJob") {

    val ppca = PPCA(3, 1.0, 1.0)

    val test_meanVector = ppca.getMean(rowMat)
    assert(test_meanVector.asBreeze === mean.asBreeze)
  }

  test("Fnormjob") {
    val br_ym = sc.broadcast(mean)
    val ppca = PPCA(3, 1.0, 1.0)
    val test_norm = ppca.fNormJob(rowMat, br_ym)
    assert(test_norm === norm)
  }

  test("YtXXtXjob") {
    val br_CM = sc.broadcast(CM)
    val br_ym = sc.broadcast(mean)
    val br_xm = sc.broadcast(xm)
    val ppca = PPCA(3, 1.0, 1.0)
    val (test_centralYtX, test_centralXtX) = ppca.YtXJob(rowMat, br_CM, br_ym, br_xm)
    assert(test_centralYtX.asBreeze === YtX.asBreeze)
    assert(test_centralXtX.asBreeze === XtX.asBreeze)
  }

  test("ss3job") {
    val br_CM = sc.broadcast(CM)
    val br_C = sc.broadcast(C)
    val ppca = PPCA(3, 1.0, 1.0)
    val test_ss3 = ppca.ss3Job(rowMat, mean, br_CM, br_C)
    assert(test_ss3 === 1.5)
  }

  test("reConstruct") {
    val br_CM = sc.broadcast(new DenseMatrix(nCols.toInt, d,
      Array(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)))
    val br_C = sc.broadcast(C)
    val br_ym = sc.broadcast(Vectors.dense(Array(1.0, 4.0, 7.0)).toDense)
    @transient val ppca = PPCA(3, 1.0, 1.0)
    val xm_v = new DenseVector(Array(2.0, 8.0, 14.0))
    val test_error = ppca.reConstruct(rowMat, xm_v, br_CM, br_C, br_ym)
    assert(test_error === 1.0)
  }

  test("reConstruct_max_error") {
    val v1 = new SparseVector(2, Array(0, 1), Array(1.0, 1.0))
    val v2 = new SparseVector(2, Array(0, 1), Array(2.0, 2.0))
    val rows = Seq(v1, v2)
    val rowMatrix = new RowMatrix(sc.parallelize(rows))
    val br_CM = sc.broadcast(new DenseMatrix(1, 2,
      Array(2.0, 0.0)))
    val br_C = sc.broadcast(new DenseMatrix(2, 1,
      Array(2.0, 2.0)))
    val br_ym = sc.broadcast(Vectors.dense(Array(0.0, 0.0)).toDense)
    val ppca = PPCA(1, 1.0, 1.0)
    val xm_v = new DenseVector(Array(0.0))
    val test_error = ppca.reConstruct_max_error(rowMatrix, xm_v, br_CM, br_C, br_ym)
    assert(test_error === 12.0)
  }

  test("ppca") {

    val v1 = new SparseVector(2, Array(0, 1), Array(1.0, 2.0))
    val v2 = new SparseVector(2, Array(0, 1), Array(3.0, 4.0))
    val v3 = new SparseVector(2, Array(0, 1), Array(5.0, 6.0))
    val rows = Seq(v1, v2, v3)
    val rowMatrix = new RowMatrix(sc.parallelize(rows))

    val ss_init = 1

    val ym = new DenseVector(Array(3.0, 4.0))

    val ss1: Double = 16.0

    val ncol = rowMatrix.numCols().toInt

    val k: Int = 1

    val CMatrix_init = new DenseMatrix(ncol, k,
      Array(1.0, 0.0))

    val ppca = PPCA(k, 1, 0.0001f, 1.0, 1.0)

    val C = ppca.computePCA(rowMatrix, ym, ss1, ss_init, CMatrix_init)
    val C_1 = new DenseMatrix(ncol, k, Array(8.0/7.0, 8.0/7.0))

    assert(C.asBreeze === C_1.asBreeze)
    // ss = 23/12
    // ss2 = 7/2
    // ss3 = 4
  }

//  test("compare") {
//    val A = new DenseMatrix(3, 3, Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
//    val B = new DenseMatrix(3, 3, Array(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0))
//    val diff = PPCA.compare(A, B)
//    assert(diff === ArrayBuffer[Double](1.0, 1.0, 1.0))
//    print(s"diffArray = $diff\n")
//  }

//  test("getU") {
//
//    val ppca = PPCA(1, 1.0, 1.0)
//
//    val C = ppca.computePCA(mat)
//    print(s"C = \n$C\n")
//
//    val U = PPCA.getU(C)
//    print(s"U = \n$U\n")
//
//    val UtU = DenseMatrix.zeros(C.numCols, C.numCols)
//
//    BLAS.gemm(1.0, U.transpose, U, 1.0, UtU)
//
//    print(s"U = $U")
//
//    val I = Matrices.eye(1)
//
//    //      assert(UtU.asBreeze === I.asBreeze)
//
//  }

//  test("getUold") {
//
//    val W = new DenseMatrix(nCols.toInt, d, Array(0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0))
//
//    val ppca = PPCA(3, 1.0, 1.0)
//
//    val x = PPCA.getU(W)
//
//    val U = Matrices.dense(nCols.toInt, d,
//      Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
//
//    assert(x.asBreeze === U.asBreeze)
//
//  }

//  test("compare") {
//
//    val X = new DenseMatrix(3, 2, Array(2, 4, 6, 4.0, 5.0, 6.0), false)
//
//    val Y = new DenseMatrix(3, 2, Array(1, 2, 3, 40, 50, 60), false)
//
//    val cosin = PPCA.compare(X, Y)
//    assert(cosin === Array(1.0, 1.0))
//  }
//
//  test("compare_max") {
//
//    val X = new DenseMatrix(3, 2, Array(2, 4, 6, 4.0, 5.0, 6.0), false)
//
//    val Y = new DenseMatrix(3, 2, Array(40, 50, 60, 1, 2, 3), false)
//
//    val cosin_max = PPCA.compare_max(X, Y)
//    assert(cosin_max === Array(1.0, 1.0))
//  }


//  test("PPCA") {
//    val ppca = PPCA(3, 1.0, 1.0)
//    ppca.computePCA(mat)
//  }



  /*
    test("projection") {
      val ppca = new PPCA(mat, 3, sc)
      val U = Matrices.dense(nCols.toInt, d,
        Array(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0))
  //    ppca.projection(U).rows.foreach(yi => print(yi.toArray.mkString(", ") + "\n"))
      val Y = ppca.projection(U)
    }



    */

}
