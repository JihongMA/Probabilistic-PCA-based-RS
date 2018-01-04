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
import scala.util.Random

import breeze.linalg.{diag, eigSym, inv, svd, trace}
import breeze.linalg.eigSym.EigSym
import org.slf4j.LoggerFactory

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.util.{AccumulatorContext, AccumulatorV2}

private [mllib] class VectorAccumulator(nCols: Int, vec: DenseVector)
    extends AccumulatorV2[Vector, Vector] {
  require( vec.size == nCols)

  private val vectorAcc = vec

  override def isZero: Boolean = vectorAcc.numNonzeros == 0

  def this(nCols: Int) {
    this (nCols, Vectors.zeros(nCols).toDense)
  }

  override def reset(): Unit = {
    vectorAcc.update((v: Double) => 0)
  }

  override def add(vec: Vector): Unit = {
    BLAS.axpy(1.0, vec, vectorAcc)
  }

  def addabs(vec: Vector): Unit = {
    vec.update(math.abs(_))
    BLAS.axpy(1.0, vec, vectorAcc)
  }

  override def merge(other: AccumulatorV2[Vector, Vector]) : Unit = other match {
    case o: VectorAccumulator =>
      BLAS.axpy(1.0, o.value, vectorAcc)
    case _ =>
      throw new IllegalArgumentException(
        s"merge of VectoroAccumulator got type ${other.getClass}.")
  }

  override def copy() : VectorAccumulator = {
    new VectorAccumulator(nCols, vectorAcc.copy)
  }

  override def value: DenseVector = vectorAcc
}


private [mllib] class MatrixAccumulator(nRows: Int, nCols: Int, matrixA: DenseMatrix)
    extends AccumulatorV2[Matrix, Matrix] {
  require(matrixA.numRows == nRows && matrixA.numCols == nCols)

  private val matrixAcc: DenseMatrix = matrixA

  override def isZero: Boolean = matrixAcc.numNonzeros == 0

  def this(nRows: Int, nCols: Int) {
    this(nRows, nCols, DenseMatrix.zeros(nRows, nCols))
  }

  override def reset(): Unit = {
    matrixAcc.update((i: Double) => 0)
  }

  override def add(mat: Matrix): Unit = {
    mat match {
      case mat_d: DenseMatrix =>
        BLAS.axpy(1.0, mat_d, matrixAcc)
      case mat_s: SparseMatrix =>
        BLAS.axpy(1.0, mat_s.toDense, matrixAcc)
    }
  }

  override def merge(other: AccumulatorV2[Matrix, Matrix]): Unit = other match {
    case o: MatrixAccumulator => BLAS.axpy(1.0, o.value, matrixAcc)
    case _ =>
      throw new IllegalArgumentException(
        s"MatrixAccumulator merge got type ${other.getClass}.")
  }

  override def copy(): MatrixAccumulator = {
    new MatrixAccumulator(nRows, nCols, matrixAcc.copy)
  }

  override def value: DenseMatrix = matrixAcc

}


 /**
  * A feature transformer that projects vectors to a low-dimensional space using PPCA
  * based on paper of T. Elgamal, M. Yabandeh, A. Aboulnaga, W. Mustafa, and M. Hefeeda.
  * sPCA: Scalable Principal Component Analysis fo Big Data on Distributed Platforms.
  * In Proc. of ACM SIGMOD15
  */
class PPCA(k: Int, maxIter: Int, threshold: Float,
     errorThreshold: Double, errorRate: Double) {
  require(k > 0, s"Number of principal components must be positive but got ${k}")
  require(maxIter> 0, s"Number of max iterations must be positive but got ${maxIter}")
  require(threshold >= 0.0,
    s"errorThreshold must be greater than zero but got ${threshold} " )

  @transient val log = LoggerFactory.getLogger(classOf[PPCA])


  // Frobenious Norm Job : Obtain Frobenius norm of the input
  def fNormJob (rowMatrix: RowMatrix, br_ym: Broadcast[DenseVector]): Double = {
    var meanSquareSum: Double = 0
    val br_ym_v: DenseVector = br_ym.value
    val sc = rowMatrix.rows.context

    br_ym_v.foreachActive( (i, v) => meanSquareSum += math.pow(v, 2))

    val doubleAccumNorm2 = sc.doubleAccumulator
    val br_meanSqSum = sc.broadcast(meanSquareSum)

    rowMatrix.rows.foreachPartition {
      var norm2: Double = 0.0
      val mean: Double = 0.0
      rowIter: Iterator[Vector] => {
        while (rowIter.hasNext) {
          val yi = rowIter.next
          yi.foreachActive((i, v) => {
            norm2 += math.pow((v - br_ym_v(i)), 2)
            norm2 -= math.pow(br_ym_v(i), 2)
          })
          norm2 += br_meanSqSum.value
        }
        doubleAccumNorm2.add(norm2)
      }
    }
    val accumNorm2 = doubleAccumNorm2.value

    // cleanup broadcast variable and accumulator
    br_meanSqSum.unpersist()
    AccumulatorContext.remove(doubleAccumNorm2.id)

    accumNorm2
  }

  /**
   * X'X and Y'X Job:  which computes two matrices X'X and Y'X
   * Xc = Yc * MEM (MEM is the in-memory broadcasted matrix Y2X)
   *
   * XtX = Xc' * Xc
   *
   * YtX = Yc' * Xc
   *
   * It also considers that Y is sparse and receives the mean vectors Ym and Xm
   * separately.
   *
   * Yc = Y - Ym
   *
   * Xc = X - Xm
   *
   * Xc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = X - Xm
   *
   * XtX = (X - Xm)' * (X - Xm)
   *
   * YtX = (Y - Ym)' * (X - Xm)
   *
   */
  def YtXJob (rowMat: RowMatrix, br_CM: Broadcast[DenseMatrix],
      br_ym: Broadcast[DenseVector],
      br_xm: Broadcast[DenseVector]): (DenseMatrix, DenseMatrix) = {

    val nCols = rowMat.numCols().toInt
    var matrixAccumXtx = new MatrixAccumulator(k, k)
    var matrixAccumYtx = new MatrixAccumulator(nCols, k)
    var vectorAccumX = new VectorAccumulator(k)

    val sc = rowMat.rows.context

    sc.register(matrixAccumXtx)
    sc.register(matrixAccumYtx)
    sc.register(vectorAccumX)

    rowMat.rows.foreachPartition {

      val internalSumYtX = DenseMatrix.zeros(nCols, k)
      val internalSumXtX = DenseMatrix.zeros(k, k)
      val internalSumX = Vectors.zeros(k).toDense
      val resX = Vectors.zeros(k).toDense
      val br_CM_v: DenseMatrix = br_CM.value
      val br_xm_v: DenseVector = br_xm.value
      rowIter: Iterator[Vector] => {

        while (rowIter.hasNext) {
          val yi = rowIter.next().toSparse

          // Perform in-memory matrix multiplication
          // xi = CM' * yi
          // resX = xi - xm
          BLAS.gemv (1.0, br_CM_v, yi, 1.0, resX)
          BLAS.axpy(-1.0, br_xm_v, resX)

           /**
           * Compute matrix YtX =  yi' * resX
           */
          val yiToMatrix = new SparseMatrix(yi.size, 1, Array(0,
            yi.values.length), yi.indices, yi.values)
          val xiToMatrix = new DenseMatrix(1, resX.size, resX.values)
          BLAS.gemm (1.0, yiToMatrix, xiToMatrix, 1.0, internalSumYtX)

          /**
           * compute matrix XtX = resX' * resX
           */
          BLAS.syr (1.0, resX, internalSumXtX)
          BLAS.axpy (1.0, resX, internalSumX)

          // reset resX
          resX.update((i) => 0)
        }
      }
      vectorAccumX.add(internalSumX)
      matrixAccumXtx.add(internalSumXtX)
      matrixAccumYtx.add(internalSumYtX)
    }

    val centralYtX = matrixAccumYtx.value
    val centralXtX = matrixAccumXtx.value

    // centralSumX = (Sum(Xi)-N*Xm)
    val centralSumX = vectorAccumX.value

    /**
     * update YtX = YtX - Ym' * centralSumX
     */
    val ymtoMatrix = new DenseMatrix(br_ym.value.size, 1, br_ym.value.values)
    val sumXtoMatrix = new DenseMatrix(1, centralSumX.size, centralSumX.values)
    BLAS.gemm(-1.0, ymtoMatrix, sumXtoMatrix, 1.0, centralYtX)

    // clean up accumulators
    AccumulatorContext.remove(matrixAccumXtx.id)
    AccumulatorContext.remove(matrixAccumYtx.id)
    AccumulatorContext.remove(vectorAccumX.id)
    matrixAccumXtx = null
    matrixAccumYtx = null
    vectorAccumX = null


    (centralYtX, centralXtX)
  }

   /**
    * xcty = Sum (xi * C' * yi')
    *
    * We also regenerate xi on demand by the following formula:
    *
    * xi = yi * y2x
    *
    * To make it efficient for sparse matrices that are not mean-centered, we receive the mean
    * separately:
    *
    * xi = (yi - ym) * y2x = yi * y2x - xm, where xm = ym*y2x
    *
    * xi * C' * (yi-ym)' = xi * ((yi-ym)*C)' = xi * (yi*C - ym*C)'
    *
    */
  def ss3Job (rowMat: RowMatrix,
      meanVector: DenseVector,
      br_CM: Broadcast[DenseMatrix],
      br_C: Broadcast[DenseMatrix]) : Double = {

    val resYmC = Vectors.zeros(k).toDense
    BLAS.gemv(1.0, br_C.value.transpose, meanVector, 1.0, resYmC)

    val br_YmC = rowMat.rows.context.broadcast(resYmC)
    val doubleAccumXctyt = rowMat.rows.context.doubleAccumulator

    // xcty = Sum (xi * C' * yi')
    // xi = yi * y2x
    // xi = (yi - ym) * y2x = yi * y2x - xm, where xm = ym*y2x
    // xi * C' * (yi-ym)' = xi * ((yi-ym)*C)' = xi * (yi*C - ym*C)'
    rowMat.rows.foreachPartition {

      val resArrayYiC = Vectors.zeros(k).toDense
      val resX = Vectors.zeros(k).toDense
      val br_C_v = br_C.value
      val br_CM_v = br_CM.value
      var internalSum: Double = 0.0

      rowIter: Iterator[Vector] => {

        while (rowIter.hasNext) {

          val yi = rowIter.next().toSparse

          BLAS.gemv(1.0, br_CM_v, yi, 1.0, resX)

          // resArrayYiC = yiS.multiplyMatrix(br_C.value.transpose)
          BLAS.gemv(1.0, br_C_v.transpose, yi, 1.0, resArrayYiC)
          // resArrayYiC = resArrayYiC - resArrayYmC
          BLAS.axpy(-1.0, br_YmC.value, resArrayYiC)
          // dotRes: resX * resArrayYiC
          internalSum += BLAS.dot(resX, resArrayYiC)

          // reset resX, resArrayYic
          resX.update((i) => 0.0)
          resArrayYiC.update((i) => 0.0)
        }
      }
      doubleAccumXctyt.add(internalSum)
    }
    val ss3 = doubleAccumXctyt.value

    // cleanup accumulator and broadcast variable
    br_YmC.unpersist(true)
    AccumulatorContext.remove(doubleAccumXctyt.id)

    ss3
  }

  /**
   * Reconstruction Error Job: The job computes the reconstruction error of the
   * input matrix after updating the principal components matrix
   * Xc = Yc * Y2X
   *
   * ReconY = Xc * C'
   *
   * Err = ReconY - Yc
   *
   * Norm2(Err) = abs(Err).zSum().max()
   * To take the sparse matrix into account we receive the mean separately:
   *
   * X = (Y - Ym) * Y2X = X - Xm, where X=Y*Y2X and Xm=Ym*Y2X
   *
   * ReconY = (X - Xm) * C' = X*C' - Xm*C'
   *
   * Err = X*C' - Xm*C' - (Y - Ym) = X*C' - Y - Zm, where where Zm = Xm*C' - Ym
   *
   */
  def reConstruct(rowMat: RowMatrix,
      xm: DenseVector,
      br_CM: Broadcast[DenseMatrix],
      br_C : Broadcast[DenseMatrix],
      br_ym: Broadcast[DenseVector]) : Double = {

    val nCols: Int = rowMat.numCols().toInt
    val resArrayZm = Vectors.zeros(nCols).toDense
    BLAS.gemv(1.0, br_C.value, xm, 1.0, resArrayZm)
    BLAS.axpy(-1.0, br_ym.value, resArrayZm)

    val sc = rowMat.rows.context
    val vectorAccumErr = new VectorAccumulator(nCols)
    val vectorAccumNormCentralized = new VectorAccumulator(nCols)
    sc.register(vectorAccumErr)
    sc.register(vectorAccumNormCentralized)
    val er = errorRate
    val br_err = sc.broadcast(er)

    rowMat.rows.foreachPartition {

      val rand = new Random()
      val xiCt = Vectors.zeros(nCols).toDense
      val resX = Vectors.zeros(k).toDense
      val ymNeg = br_ym.value.copy
      BLAS.scal(-1.0, ymNeg)

      rowIter: Iterator[Vector] => {

        while (rowIter.hasNext) {

          val yi = rowIter.next().toSparse

          val nextRand = rand.nextDouble
          if (nextRand < br_err.value) {

            // resX = br_CM' * yi
            BLAS.gemv(1.0, br_CM.value, yi, 1.0, resX)

            // XiCt = br_C' * resX
            BLAS.gemv(1.0, br_C.value, resX, 1.0, xiCt)
            // reconstructedError = xiCt - yi - resArrayZm
            BLAS.axpy(-1.0, yi, xiCt)
            BLAS.axpy(-1.0, resArrayZm, xiCt )
            vectorAccumErr.addabs(xiCt)

            val normalizedArray = ymNeg.copy

            BLAS.axpy(1.0, yi, normalizedArray)
            vectorAccumNormCentralized.addabs(normalizedArray)

            // reset resX, xiCt
            resX.update((i) => 0.0)
            xiCt.update((i) => 0.0)
          }
        }
      }
    }

    val reconstructionError = vectorAccumErr.value(vectorAccumErr.value.argmax)

    log.info("************************ReconsructionError=" + reconstructionError)

    val centralizedYNorm = vectorAccumNormCentralized.value(vectorAccumNormCentralized.value.argmax)

    log.info("************************CentralizedNOrm=" + centralizedYNorm)

    // cleanup broadcast variable and accumulators
    AccumulatorContext.remove(vectorAccumErr.id)
    AccumulatorContext.remove(vectorAccumNormCentralized.id)
    br_err.unpersist()

    reconstructionError/centralizedYNorm
  }

   def reConstruct_max_error(rowMat: RowMatrix,
                   xm: DenseVector,
                   br_CM: Broadcast[DenseMatrix],
                   br_C : Broadcast[DenseMatrix],
                   br_ym: Broadcast[DenseVector]) : Double = {

     val nCols: Int = rowMat.numCols().toInt
     val resArrayZm = Vectors.zeros(nCols).toDense
     BLAS.gemv(1.0, br_C.value, xm, 1.0, resArrayZm)
     BLAS.axpy(-1.0, br_ym.value, resArrayZm)

     val sc = rowMat.rows.context
     val er = errorRate
     val br_err = sc.broadcast(er)

     val max_error_array = rowMat.rows.mapPartitions{
       var max: Double = 0.0
       val rand = new Random()
       val xiCt = Vectors.zeros(nCols).toDense
       val resX = Vectors.zeros(k).toDense
       val ymNeg = br_ym.value.copy
       BLAS.scal(-1.0, ymNeg)

       rowIter: Iterator[Vector] => {

         while (rowIter.hasNext) {

           val yi = rowIter.next().toSparse

           val nextRand = rand.nextDouble
           if (nextRand < br_err.value) {

             // resX = br_CM' * yi
             BLAS.gemv(1.0, br_CM.value, yi, 1.0, resX)

             // XiCt = br_C' * resX
             BLAS.gemv(1.0, br_C.value, resX, 1.0, xiCt)
             // reconstructedError = xiCt - yi - resArrayZm
             BLAS.axpy(-1.0, yi, xiCt)
             BLAS.axpy(-1.0, resArrayZm, xiCt )


             if(Vectors.norm(xiCt, 1) > max ) {
               max = Vectors.norm(xiCt, 1)
             }

             // reset resX, xiCt
             resX.update((i) => 0.0)
             xiCt.update((i) => 0.0)
           }
         }
         Iterator(max)
       }
     }.collect()

     log.info("************************max_reconstruction_error=" + max_error_array.max)

     // cleanup broadcast variable and accumulators
     br_err.unpersist()
     max_error_array.max

   }

   def getMean (rowMat: RowMatrix) : Vector = {

     val meanVector = rowMat.computeColumnSummaryStatistics().mean
     //    log.info("ym = " + meanVector.toDense.toArray.mkString(", "))
     meanVector
   }

   def guess_CMatrix (rowMatrix: RowMatrix, meanVector:
    DenseVector, sample_rate: Double) : DenseMatrix = {
     val sc = rowMatrix.rows.context

     val nCols = rowMatrix.numCols().toInt

     val br_sample_rate = sc.broadcast(sample_rate)

     val rand = new Random()
     var sample_count: Long = 0

     val sampleRows = rowMatrix.rows.mapPartitions {

       var samples = List[Vector]()
       rowIter: Iterator[Vector] => {
         while (rowIter.hasNext) {
           val yi = rowIter.next().toSparse

           val nextRand = rand.nextDouble
           if (nextRand < br_sample_rate.value) {
             sample_count += 1
             samples :+= yi
           }
         }
       }
         Iterator(samples)
     }.flatMap(l => l)

     val sampleMatrix = new RowMatrix(sampleRows, sample_count, nCols)

     computePCA(sampleMatrix, meanVector)
   }

   /**
    * Compute principal components
    */
   def computePCA(rowMatrix: RowMatrix): DenseMatrix = {
     val meanVector = getMean(rowMatrix).toDense
     computePCA(rowMatrix, meanVector)
   }

  def computePCA(rowMatrix: RowMatrix, meanVector: DenseVector): DenseMatrix = {
    val br_ym = rowMatrix.rows.context.broadcast(meanVector)
    // compute Frobenious Norm
    val ss1: Double = fNormJob(rowMatrix, br_ym)
    val ss_init = Random.nextDouble()
    val rng = Random.self
    val nCols = rowMatrix.numCols()
    val CMatrix_init = Matrices.randn(nCols.toInt, k, rng) match {
      case dm: DenseMatrix => dm
      case sm: SparseMatrix => sm.toDense
    }
    val pcaMatrix = computePCA(rowMatrix, br_ym, ss1, ss_init, CMatrix_init)

    br_ym.unpersist(true)

    pcaMatrix
  }

  def computePCA(rowMatrix: RowMatrix, meanVector: DenseVector, ss1: Double,
                 ss_init: Double, CMatrix_init: DenseMatrix): DenseMatrix = {
    val br_ym = rowMatrix.rows.context.broadcast(meanVector)

    val pcaMatrix = computePCA(rowMatrix, br_ym, ss1, ss_init, CMatrix_init)

    br_ym.unpersist(true)

    pcaMatrix
  }

  private def computePCA(rowMatrix: RowMatrix, br_ym: Broadcast[DenseVector],
      ss1: Double, ss_init: Double, CMatrix_init: DenseMatrix): DenseMatrix = {

    log.info("PPCA computePrincipalComponents")

    val nRows = rowMatrix.numRows()

    val nCols = rowMatrix.numCols()

    log.info(s"Rows = $nRows, Cols = $nCols")

//    log.info(s"Mean = ${br_ym.value}")

//    log.info("NOOOOORM2=" + ss1);

    val sc = rowMatrix.rows.context

    var round: Int = 0
    var prevObjective: Double = Double.MaxValue
    var error: Double = 0
    var relChangeInObjective: Double = Double.MaxValue
    var prevError: Double = Double.MaxValue
    val meanVector = br_ym.value
    val CMatrix = CMatrix_init
    var ss = ss_init

    val cTc: DenseMatrix = DenseMatrix.zeros(k, k)
    BLAS.gemm(1.0, CMatrix.transpose, CMatrix, 1.0, cTc)
    val cm = DenseMatrix.zeros(k, nCols.toInt)
    val xtxC = DenseMatrix.zeros(k, k)
    val xm = Vectors.zeros(k) match {
      case dv: DenseVector => dv
      case sv: SparseVector => sv.toDense
    }

    val errorArray = new ArrayBuffer[Double]()
    val changeArray = new ArrayBuffer[Double]()

    while (round < maxIter && relChangeInObjective > threshold &&
      prevError > errorThreshold) {
      // print(s"oldCMatrix = \n$CMatrix\n")

      // compute CCt = C' * C + ss * I
      BLAS.axpy(ss, DenseMatrix.eye(k), cTc)
      // log.info("CCt = \n" + CCt.toString)

      val MInv = Matrices.fromBreeze(inv(cTc.asBreeze.toDenseMatrix)) match {
        case dm: DenseMatrix => dm
        case sm: SparseMatrix => sm.toDense
      }

      // compute CM = C * ccT'
      BLAS.gemm(1.0, MInv, CMatrix.transpose, 1.0, cm)
      // log.info("CM = \n" + cm.toString)

      // print(s"cm = \n$cm\n")
      // Broadcast CM
      val br_CM = sc.broadcast(cm)

      // compute XM = meanVector * CM
      BLAS.gemv(1.0, cm, meanVector, 1.0, xm)

      // Broadcast Xm because it will be used in several iterations.
      val br_xm = sc.broadcast(xm)
      // log.info(s"xm = $xm")

      // print(s"xm = \n${br_xm.value}\n")
      // print(s"ym = \n${br_ym.value}\n")

      // ---------------------------------- YtX XtX job---------------------------------
      val (centralYtX, centralXtX) = YtXJob(rowMatrix, br_CM, br_ym, br_xm)
      // ---------------------------------- End of YtX XtX job----------------------------
      // log.info(s"centralYtX = \n $centralYtX \n, centralXtX = \n $centralXtX")

      // print(s"centralYtX = \n$centralYtX\n")

      // Xtx += ss * ccTInv
      BLAS.axpy(nRows * ss, MInv, centralXtX)
      // C = (Ye'*X) / SumXtX;
      // inv(DenseMatrix)
      val invXtX_central = Matrices.fromBreeze(inv(centralXtX.asBreeze.toDenseMatrix)) match {
        case dm: DenseMatrix => dm
        case sm: SparseMatrix => sm.toDense
      }
      // log.info(s"invXtX_central = \n $invXtX_central")
      CMatrix.update((i) => 0.0)
      // update CMatrix = YtX * invXtX
      BLAS.gemm(1.0, centralYtX, invXtX_central, 1.0, CMatrix)

      val br_C = sc.broadcast(CMatrix)

      cTc.update((i) => 0.0)

      BLAS.gemm(1.0, CMatrix.transpose, CMatrix, 1.0, cTc)

      BLAS.gemm(1.0, centralXtX, cTc, 1.0, xtxC)

      val ss2 = trace(xtxC.asBreeze.toDenseMatrix)

      // ---------------------------------- ss3Job----------------------------
      val ss3 = ss3Job(rowMatrix, meanVector, br_CM, br_C)
      // ---------------------------------- End of ss3Job----------------------------

      ss = (ss1 + ss2 - 2 * ss3) / (nRows * nCols)

      error = reConstruct_max_error(rowMatrix, xm, br_CM, br_C, br_ym)

      val objective = error
      relChangeInObjective = Math.abs(1 - objective / prevObjective)
      prevObjective = objective

      log.info("Computing the error at round " + round + " ...")
      log.info(s"*********Relative change: $relChangeInObjective")
      log.info("... end of computing the error at round " + round + " And error=" + error)
      log.info(s"***********ss = : $ss")

      prevError = error

      round += 1

      errorArray += error
      changeArray += relChangeInObjective

      // reset matrix
//      cTc.update((i) => 0.0)
//      cTc.updateEye(1.0)
      cm.update((i) => 0.0)
      xm.update((i) => 0.0)
      xtxC.update((i) => 0.0)

      // cleanup broadcast variable
      br_CM.unpersist(true)
      br_xm.unpersist(true)
      br_C.unpersist(true)
    }
    // print(s"errors = $errorArray\n")
    log.info("errors = " + errorArray)
    log.info("rel Change = " + changeArray)
    // print(s"newCMatrix = \n$CMatrix\n")
    // print(s"*******newss = $ss\n")
    CMatrix
  }

}

 /**
  * Factory methods for PPCA.
  */

object PPCA{
   def apply(k: Int, maxIter: Int, threshold: Float,
             errorThreshold: Double, errorRate: Double): PPCA = {
     new PPCA(k, maxIter, threshold, errorThreshold, errorRate)
   }

   def apply(d: Int, error: Double, sampleRate: Double): PPCA = {
     new PPCA(d, 10, 0.0001f, error, sampleRate)
   }

   def svdC(C : DenseMatrix): DenseMatrix = {
     val svd.SVD(u, s, v) = svd.reduced(C.asBreeze.toDenseMatrix)
     val U = Matrices.fromBreeze(u) match {
       case dm: DenseMatrix => dm
       case sm: SparseMatrix => sm.toDense
     }
     // print(s"org: \n$U\n")
     val iter = U.colIter
     var j = 0
     while(iter.hasNext) {
       val v = iter.next()
       val vabs = v.copy
       vabs.update(x => math.abs(x))
       val maxidx = vabs.argmax
       val sign = v(maxidx).signum
       v.foreachActive((i, x) => U.update(i, j, x * sign))
       j = j + 1
       // print(s"sign = $sign , max = ${v(maxidx); }")
     }
     // print(s"sign converted: \n$U\n")
     U
   }

   def getU(C : DenseMatrix): DenseMatrix = {

     val CtC = DenseMatrix.zeros(C.numCols, C.numCols)

     BLAS.gemm(1.0, C.transpose, C, 1.0, CtC)

     val EigSym(lambda, evs) = eigSym(CtC.asBreeze.toDenseMatrix)

     val R = evs

     val Rt = Matrices.fromBreeze(evs.t) match {
       case dm: DenseMatrix => dm
       case sm: SparseMatrix => sm.toDense
     }

     val L = lambda
     val size = lambda.length
     var i = 0
     while (i < size) {
       val sigma = L(i)
       L.update(i, 1/math.sqrt(sigma))
       i += 1
     }
     val lt = Vectors.fromBreeze(L.toDenseVector).toDense

     val tMatrix: DenseMatrix = DenseMatrix.zeros(Rt.numRows, lt.size)
     BLAS.gemm(1.0, Rt, DenseMatrix.diag(lt), 1.0, tMatrix)
     val uMatrix: DenseMatrix = DenseMatrix.zeros(C.numRows, lt.size)
     BLAS.gemm(1.0, C, tMatrix, 1.0, uMatrix)
     uMatrix
   }

   def compare(result: DenseMatrix, other: DenseMatrix): Array[Double] = {
     require(result.numCols == other.numCols & result.numRows == other.numRows)
     val diffArray = ArrayBuffer[Double]()
     val iter = result.colIter.zip(other.colIter)
     iter.foreach(pairVec => diffArray +=
       BLAS.dot(pairVec._1, pairVec._2)/
         (Vectors.norm(pairVec._1, 2) * (Vectors.norm(pairVec._2, 2))))
     diffArray.toArray
   }

   def compare_max(result: DenseMatrix, other: DenseMatrix): Array[Double] = {
     require(result.numCols == other.numCols & result.numRows == other.numRows)
     val diffArray = ArrayBuffer[Double]()
     val iter1 = result.colIter
     val visted = ArrayBuffer[Int]()

     while(iter1.hasNext) {
       var max = -1.0
       var max_index = 0
       var i = 0
       val vec1 = iter1.next()
       val iter2 = other.colIter
       while(iter2.hasNext) {
         val vec2 = iter2.next()
         if(!visted.contains(i)) {
           val cosin = BLAS.dot(vec1, vec2)/ (Vectors.norm(vec1, 2) * (Vectors.norm(vec2, 2)))
           print(s"cosin = $cosin\n")
           if (cosin > max) {
             max = cosin
             max_index = i
           }
         }
         i += 1
       }
       diffArray += max
       visted += max_index
       // print(s"max = $max\n")
       // print(s"index = $max_index\n")
     }
     diffArray.toArray
   }
}

 /**
  * Model fitted by [[PPCA]] that can project vectors to a low-dimensional space using PCA.
  *
  * @param k number of principal components.
  * @param pc a principal components Matrix. Each column is one principal component.
  */
class PPCAModel private[spark] (
                                val k: Int,
                                val pc: DenseMatrix,
                                val explainedVariance: DenseVector) extends VectorTransformer {
   /**
    * Transform a vector by computed Principal Components.
    *
    * @param vector vector to be transformed.
    *               Vector must be the same length as the source vectors given to `PCA.fit()`.
    * @return transformed vector. Vector will be of length k.
    */
  override def transform(vector: Vector): Vector = {
    vector match {
      case dv: DenseVector =>
        pc.transpose.multiply(dv)
      case SparseVector(size, indices, values) =>
        /* SparseVector -> single row SparseMatrix */
        val sm = Matrices.sparse(size, 1, Array(0, indices.length), indices, values).transpose
        val projection = sm.multiply(pc)
        Vectors.dense(projection.values)
      case _ =>
        throw new IllegalArgumentException("Unsupported vector format. Expected " +
          s"SparseVector or DenseVector. Instead got: ${vector.getClass}")
    }
  }
}







