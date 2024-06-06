/*
 * Copyright 2015 University of Basel, Graphics and Vision Research Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package scalismo.statisticalmodel

import breeze.linalg.svd.SVD
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.Gaussian
import scalismo.common.*
import scalismo.common.DiscreteField.vectorize
import scalismo.common.interpolation.{FieldInterpolator, NearestNeighborInterpolator}
import scalismo.geometry.*
import scalismo.kernels.{DiscreteMatrixValuedPDKernel, MatrixValuedPDKernel}
import scalismo.numerics.{GramDiagonalize, PivotedCholesky, Sampler}
import scalismo.statisticalmodel.DiscreteLowRankGaussianProcess.{Eigenpair as DiscreteEigenpair, *}
import scalismo.statisticalmodel.LowRankGaussianProcess.Eigenpair
import scalismo.statisticalmodel.NaNStrategy.NanIsNumericValue
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.{Memoize, Random}

import scala.collection.parallel.immutable.ParVector
import scala.language.higherKinds

/**
 * Represents a low-rank gaussian process, that is only defined at a finite, discrete set of points. It supports the
 * same operations as the LowRankGaussianProcess class, but always returns instead a discrete representation.
 * Furthermore, most operations are much more efficient, as they are implemented using fast matrix/vector operations.
 *
 * Where the modeled functions in a LowRankGaussianProcess are of type Point[D]=>Vector[D], this discretized version is
 * of type VectorPointData.
 *
 * It is possible to convert a DiscreteLowRankGaussianProcess to a LowRankGaussianProcess by calling the interpolation
 * method.
 *
 * @see
 *   [[scalismo.common.DiscreteField]]
 * @see
 *   [[DiscreteLowRankGaussianProcess]]
 */
class DiscreteLowRankGaussianProcess[D: NDSpace, DDomain[DD] <: DiscreteDomain[DD], Value] private[scalismo] (
  _domain: DDomain[D],
  val meanVector: DenseVector[Double],
  val variance: DenseVector[Double],
  val basisMatrix: DenseMatrix[Double]
)(implicit override val vectorizer: Vectorizer[Value])
    extends DiscreteGaussianProcess[D, DDomain, Value](
      DiscreteField.createFromDenseVector[D, DDomain, Value](_domain, meanVector),
      basisMatrixToCov(_domain, variance, basisMatrix)
    ) {
  self =>

  /** See [[DiscreteLowRankGaussianProcess.rank]] */
  val rank: Int = basisMatrix.cols

  /**
   * Discrete version of [[DiscreteLowRankGaussianProcess.instance]]
   */
  def instance(c: DenseVector[Double]): DiscreteField[D, DDomain, Value] = {
    require(rank == c.size)
    val instVal = instanceVector(c)
    DiscreteField.createFromDenseVector[D, DDomain, Value](domain, instVal)
  }

  /**
   * Get instance at a specific point
   */
  def instanceAtPoint(c: DenseVector[Double], pid: PointId): Value = {
    require(rank == c.size)
    vectorizer.unvectorize(instanceVectorAtPoint(c, pid))
  }

  /**
   * Returns the probability density of the instance produced by the x coefficients
   */
  def pdf(coefficients: DenseVector[Double]) = {
    if (coefficients.size != rank)
      throw new Exception(s"invalid vector dimensionality (provided ${coefficients.size} should be $rank)")
    val mvnormal = MultivariateNormalDistribution(DenseVector.zeros[Double](rank), diag(DenseVector.ones[Double](rank)))
    mvnormal.pdf(coefficients)
  }

  /**
   * Returns the log of the probability density of the instance produced by the x coefficients.
   *
   * If you are interested in ordinal comparisons of PDFs, use this as it is numerically more stable
   */
  def logpdf(coefficients: DenseVector[Double]) = {
    if (coefficients.size != rank)
      throw new Exception(s"invalid vector dimensionality (provided ${coefficients.size} should be $rank)")
    val mvnormal = MultivariateNormalDistribution(DenseVector.zeros[Double](rank), diag(DenseVector.ones[Double](rank)))
    mvnormal.logpdf(coefficients)
  }

  /**
   * Returns the probability density of the given instance
   */
  override def pdf(instance: DiscreteField[D, DDomain, Value]): Double = pdf(coefficients(instance))

  /**
   * Returns the log of the probability density of the instance
   *
   * If you are interested in ordinal comparisons of PDFs, use this as it is numerically more stable
   */
  override def logpdf(instance: DiscreteField[D, DDomain, Value]): Double = logpdf(coefficients(instance))

  /**
   * Discrete version of [[DiscreteLowRankGaussianProcess.sample]]
   */
  override def sample()(implicit random: Random): DiscreteField[D, DDomain, Value] = {
    val standardNormal = Gaussian(0, 1)(random.breezeRandBasis)
    val coeffs = standardNormal.sample(rank)
    instance(DenseVector(coeffs.toArray))
  }

  /**
   * Returns the variance and associated basis function that defines the process. The basis is the (discretized)
   * Karhunen Loeve basis (e.g. it is obtained from a Mercer's decomposition of the covariance function
   */
  def klBasis: KLBasis[D, DDomain, Value] = {
    for (i <- 0 until rank) yield {
      val eigenValue = variance(i)
      val eigenFunction = DiscreteField.createFromDenseVector[D, DDomain, Value](domain, basisMatrix(::, i))
      DiscreteEigenpair(eigenValue, eigenFunction)
    }
  }

  /**
   * Returns a reduced rank model, using only the leading basis function of the Karhunen-loeve expansion.
   *
   * @param newRank:
   *   The rank of the new Gaussian process.
   */
  def truncate(rank: Int) = {
    DiscreteLowRankGaussianProcess[D, DDomain, Value](mean, klBasis.take(rank))
  }

  /**
   * Discrete version of [[LowRankGaussianProcess.project(IndexedSeq[(Point[D], Vector[DO])], Double)]]
   */
  override def project(s: DiscreteField[D, DDomain, Value]): DiscreteField[D, DDomain, Value] = {
    instance(coefficients(s))
  }

  /**
   * Discrete version of [[DiscreteLowRankGaussianProcess.coefficients(IndexedSeq[(Point[D], Vector[DO], Double)])]]
   */
  def coefficients(s: DiscreteField[D, DDomain, Value]): DenseVector[Double] = {
    val sigma2 = 1e-5 // regularization weight to avoid numerical problems
    val noiseDist =
      MultivariateNormalDistribution(DenseVector.zeros[Double](outputDim), DenseMatrix.eye[Double](outputDim) * sigma2)
    val td = s.valuesWithIds.map { case (v, id) => (id, v, noiseDist) }.toIndexedSeq

    val LowRankRegressionComputation(minv, yVec, mVec, qtL) =
      LowRankRegressionComputation.fromDiscreteLowRankGP(this, td, NaNStrategy.NanIsNumericValue)
    val mean_coeffs = (minv * qtL) * (yVec - mVec)
    mean_coeffs
  }

  /**
   * Discrete version of
   * [[DiscreteLowRankGaussianProcess.posterior(IndexedSeq[(Point[D], Vector[DO])], sigma2: Double]]. In contrast to
   * this method, the points for the training data are defined by the pointId. The returned posterior process is defined
   * at the same points.
   */
  def posterior(trainingData: IndexedSeq[(PointId, Value)],
                sigma2: Double
  ): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {
    val cov =
      MultivariateNormalDistribution(DenseVector.zeros[Double](outputDim), DenseMatrix.eye[Double](outputDim) * sigma2)
    val newtd = trainingData.map { case (ptId, df) => (ptId, df, cov) }
    posterior(newtd)
  }

  /**
   * Discrete version of [[DiscreteLowRankGaussianProcess.posterior(IndexedSeq[(Point[D], Vector[DO], Double)])]]. In
   * contrast to this method, the points for the training data are defined by the pointId. The returned posterior
   * process is defined at the same points.
   */
  def posterior(
    trainingData: IndexedSeq[(PointId, Value, MultivariateNormalDistribution)]
  ): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.regression(this, trainingData)
  }

  /**
   * Discrete version of
   * [[DiscreteLowRankGaussianProcess.posteriorMean(IndexedSeq[(Point[D], Vector[DO])], sigma2: Double]]. In contrast to
   * this method, the points for the training data are defined by the pointId. The returned posterior process is defined
   * at the same points.
   */
  def posteriorMean(trainingData: IndexedSeq[(PointId, Value)], sigma2: Double): DiscreteField[D, DDomain, Value] = {
    val cov =
      MultivariateNormalDistribution(DenseVector.zeros[Double](outputDim), DenseMatrix.eye[Double](outputDim) * sigma2)
    val newtd = trainingData.map { case (ptId, df) => (ptId, df, cov) }
    posteriorMean(newtd)
  }

  /**
   * Discrete version of [[DiscreteLowRankGaussianProcess.posteriorMean(IndexedSeq[(Point[D], Vector[DO], Double)])]].
   * In contrast to this method, the points for the training data are defined by the pointId. The returned posterior
   * process is defined at the same points.
   */
  def posteriorMean(
    trainingData: IndexedSeq[(PointId, Value, MultivariateNormalDistribution)]
  ): DiscreteField[D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.regressionMean(this, trainingData)
  }

  override def marginal(pointIds: Seq[PointId])(implicit
    domainCreator: UnstructuredPoints.Create[D]
  ): DiscreteLowRankGaussianProcess[D, UnstructuredPointsDomain, Value] = {
    val domainPts = domain.pointSet.points.toIndexedSeq

    val newPts = pointIds.map(pointId => domainPts(pointId.id)).toIndexedSeq
    val newDomain = UnstructuredPointsDomain(domainCreator.create(newPts))

    val newMean =
      DiscreteField[D, UnstructuredPointsDomain, Value](newDomain, pointIds.toIndexedSeq.map(id => mean(id)))

    val newKLBasis = for (DiscreteEigenpair(lambda, phi) <- klBasis) yield {
      val newValues = pointIds.map(i => phi(i)).toIndexedSeq
      DiscreteEigenpair[D, UnstructuredPointsDomain, Value](lambda, DiscreteField(newDomain, newValues))

    }

    DiscreteLowRankGaussianProcess[D, UnstructuredPointsDomain, Value](newMean, newKLBasis)
  }

  /**
   * calculates the log marginal likelihood given trainingData.
   *
   * @param trainingData
   *   Point/value pairs where that the sample should approximate, together with an error model (the uncertainty) at
   *   each point.
   */
  override def marginalLikelihood(td: IndexedSeq[(PointId, Value, MultivariateNormalDistribution)]): Double = {
    require(td.nonEmpty, "provide observations to calculate the marginal likelihood")
    val matIds = td.flatMap(t => t._1.id * outputDim until t._1.id * outputDim + outputDim)
    val effectiveRank = variance.data.count(_ > 1e-10)
    val S = if (effectiveRank == rank) variance else variance(0 until effectiveRank)
    val U = DenseMatrix.tabulate[Double](matIds.length, effectiveRank)((x, y) => basisMatrix(matIds(x), y))
    val y = DenseVector(td.flatMap(t => (vectorizer.vectorize(t._2) - vectorizer.vectorize(mean(t._1))).toArray): _*)
    LowRankGaussianProcess.marginalLikelihoodComputation(U, S, y, td.map(_._3))
  }

  /**
   * Interpolates the gaussian process using the given interpolator. Interpolation is achieved by interoplating the mean
   * and eigenfunctions using the given interpolator.
   *
   * @param interpolator
   *   the interpolator used to interpolate the mean and eigenfunctions
   * @return
   *   a (continuous) low-rank Gaussian process
   */
  override def interpolate(interpolator: FieldInterpolator[D, DDomain, Value]): LowRankGaussianProcess[D, Value] = {
    val newKLBasis = for (DiscreteEigenpair(eigenVal, eigenFun) <- klBasis) yield {
      Eigenpair(eigenVal, eigenFun.interpolate(interpolator))
    }
    val newMean = this.mean.interpolate(interpolator)

    new InterpolatedLowRankGaussianProcess(Field(EuclideanSpace[D], newMean), newKLBasis, this, interpolator)
  }

  /**
   * Interpolates discrete Gaussian process to have a new, continuous representation as a
   * [[DiscreteLowRankGaussianProcess]]. This is achieved by using a Nystrom method for computing the kl basis. The mean
   * function is currently interpolated using a nearest neighbor approach.
   *
   * @param nNystromPoints
   *   determines how many points of the domain are used to estimate the full kl basis.
   */
  def interpolateNystrom(nNystromPoints: Int = 2 * rank)(implicit rand: Random): LowRankGaussianProcess[D, Value] = {

    val pointSet = domain.pointSet
    val sampler = new Sampler[D] {
      override def volumeOfSampleRegion = numberOfPoints.toDouble

      override val numberOfPoints = nNystromPoints
      val p = volumeOfSampleRegion / numberOfPoints

      val domainPoints = pointSet.points.toIndexedSeq

      override def sample() = {
        val sampledPtIds = for (_ <- 0 until nNystromPoints) yield rand.scalaRandom.nextInt(pointSet.numberOfPoints)
        sampledPtIds.map(ptId => (domainPoints(ptId), p))
      }
    }

    // TODO, here we could do something smarter, such as e.g. b-spline interpolation
    val meanPD = this.mean

    def meanFun(pt: Point[D]): Value = {
      val closestPtId = pointSet.findClosestPoint(pt).id
      meanPD(closestPtId)
    }

    val covFun: MatrixValuedPDKernel[D] = new MatrixValuedPDKernel[D] {
      override val domain = EuclideanSpace[D]

      override def k(x: Point[D], y: Point[D]): DenseMatrix[Double] = {
        val xId = pointSet.findClosestPoint(x).id
        val yId = pointSet.findClosestPoint(y).id
        cov(xId, yId)
      }

      override val outputDim = self.outputDim
    }
    val gp = GaussianProcess(Field(EuclideanSpace[D], meanFun _), covFun)
    LowRankGaussianProcess.approximateGPNystrom[D, Value](gp, sampler, rank)
  }

  /**
   * Interpolates discrete Gaussian process to have a new, continuous representation as a
   * [[DiscreteLowRankGaussianProcess]], using nearest neigbor interpolation (for both mean and covariance function)
   */
  @deprecated("please use the [[interpolate]] method with a [[NearestNeighborInterpolator]] instead", "0.16")
  override def interpolateNearestNeighbor: LowRankGaussianProcess[D, Value] = {

    val meanPD = this.mean

    // we cache the closest point computation, as it might be heavy for general domains, and we know that
    // we will have the same oints for all the eigenfunctions
    val findClosestPointMemo =
      Memoize((pt: Point[D]) => domain.pointSet.findClosestPoint(pt).id, cacheSizeHint = 1000000)

    def meanFun(closestPointFun: Point[D] => PointId)(pt: Point[D]): Value = {
      val closestPtId = closestPointFun(pt)
      meanPD(closestPtId)
    }

    def phi(i: Int, closetPointFun: Point[D] => PointId)(pt: Point[D]): Value = {
      val closestPtId = closetPointFun(pt)
      val slice = basisMatrix(closestPtId.id * outputDim until (closestPtId.id + 1) * outputDim, i).toDenseVector
      vectorizer.unvectorize(slice)
    }

    val interpolatedKLBasis = {

      (0 until rank) map (i => Eigenpair(variance(i), Field(EuclideanSpace[D], phi(i, findClosestPointMemo))))
    }
    new InterpolatedLowRankGaussianProcess(Field(EuclideanSpace[D], meanFun(findClosestPointMemo)),
                                           interpolatedKLBasis,
                                           this,
                                           NearestNeighborInterpolator()
    )
  }

  /**
   * realigns the model on the provided part of the domain. By default aligns over the translation and approximately
   * over rotation as well. The rotation will always be calculated around the center of the provided ids. Rotations are
   * around the cardinal directions.
   *
   * @param ids
   *   these define the parts of the domain that are aligned to
   * @param withRotation
   *   True if the rotation should be included. False makes the realignment over translation exact.
   * @param diagonalize
   *   True if a diagonal basis should be returned. False is cheaper for exclusively drawing samples.
   * @return
   *   The resulting [[DiscreteLowRankGaussianProcess]] aligned on the provided instances of [[PointId]]
   */
  def realign(ids: IndexedSeq[PointId], withRotation: Boolean = true, diagonalize: Boolean = true)(implicit
    vectorizer: Vectorizer[Value], realignmentType: Realignment[D, DDomain, Value]
  ): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.realignment(this, ids, withRotation, diagonalize)
  }

  protected[statisticalmodel] def instanceVector(alpha: DenseVector[Double]): DenseVector[Double] = {
    require(rank == alpha.size)

    basisMatrix * (stddev *:* alpha) + meanVector
  }

  protected[statisticalmodel] def instanceVectorAtPoint(alpha: DenseVector[Double],
                                                        pid: PointId
  ): DenseVector[Double] = {
    require(rank == alpha.size)
    val range = pid.id * vectorizer.dim until (pid.id + 1) * vectorizer.dim

    // copy seems to be necessary in breeze 2.0, as othewise slice does not work
    val rowsForPoint = basisMatrix(range, ::).copy
    rowsForPoint * (stddev *:* alpha) + meanVector(range)
  }

  private[this] val stddev = variance.map(x => math.sqrt(x))

  override def equals(obj: Any): Boolean = {
    obj match {
      case other: DiscreteLowRankGaussianProcess[_, _, _] =>
        (this.domain == other.domain) &&
        breeze.linalg.norm(this.meanVector - other.meanVector) < 1e-10 &&
        breeze.linalg.norm(this.variance - other.variance) < 1e-10 &&
        breeze.linalg.norm(this.basisMatrix.flatten() - other.basisMatrix.flatten()) < 1e-10
      case _ => false
    }
  }
}

/**
 * Convenience class to speedup sampling from a LowRankGaussianProcess obtained by an interpolation of a
 * DiscreteLowRankGaussianProcess
 */
private[scalismo] class InterpolatedLowRankGaussianProcess[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
  mean: Field[D, Value],
  klBasis: LowRankGaussianProcess.KLBasis[D, Value],
  discreteGP: DiscreteLowRankGaussianProcess[D, DDomain, Value],
  interpolator: FieldInterpolator[D, DDomain, Value]
)(implicit vectorizer: Vectorizer[Value])
    extends LowRankGaussianProcess[D, Value](mean, klBasis) {

  require(klBasis.size == discreteGP.rank)

  override def instance(c: DenseVector[Double]): Field[D, Value] = discreteGP.instance(c).interpolate(interpolator)

  // if all training data points belong to the interpolated discrete domain, then we compute a discrete posterior GP and interpolate it
  // this way the posterior will also remain very efficient when sampling.
  override def posterior(
    trainingData: IndexedSeq[(Point[D], Value, MultivariateNormalDistribution)]
  ): LowRankGaussianProcess[D, Value] = {

    val pointSet = discreteGP.domain.pointSet
    val allInDiscrete = trainingData.forall { case (pt, vc, nz) => pointSet.isDefinedAt(pt) }

    if (allInDiscrete) {
      val discreteTD = trainingData.map { case (pt, vc, nz) => (pointSet.findClosestPoint(pt).id, vc, nz) }
      discreteGP.posterior(discreteTD).interpolate(interpolator)
    } else {
      LowRankGaussianProcess.regression(this, trainingData)
    }
  }

}

object DiscreteLowRankGaussianProcess {

  case class Eigenpair[D, DDomain[DD] <: DiscreteDomain[DD], Value](eigenvalue: Double,
                                                                    eigenfunction: DiscreteField[D, DDomain, Value]
  )

  type KLBasis[D, DDomain[D] <: DiscreteDomain[D], Value] = Seq[Eigenpair[D, DDomain, Value]]

  /**
   * Creates a new DiscreteLowRankGaussianProcess by discretizing the given gaussian process at the domain points.
   */
  def apply[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
    domain: DDomain[D],
    gp: LowRankGaussianProcess[D, Value]
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {

    val points = domain.pointSet.points.toSeq
    val outputDim = gp.outputDim

    // precompute all the at the given points

    val m = DenseVector.zeros[Double](points.size * outputDim)
    for (xWithIndex <- new ParVector(points.toVector.zipWithIndex)) {
      val (x, i) = xWithIndex
      m((i * outputDim) until ((i + 1) * outputDim)) := vectorizer.vectorize(gp.mean(x))
    }

    val U = DenseMatrix.zeros[Double](points.size * outputDim, gp.rank)
    val lambdas = DenseVector.zeros[Double](gp.rank)
    for (xWithIndex <- new ParVector(points.zipWithIndex.toVector); (eigenPair_j, j) <- gp.klBasis.zipWithIndex) {
      val LowRankGaussianProcess.Eigenpair(lambda_j, phi_j) = eigenPair_j
      val (x, i) = xWithIndex
      val v = phi_j(x)
      U(i * outputDim until (i + 1) * outputDim, j) := vectorizer.vectorize(v)
      lambdas(j) = lambda_j
    }

    new DiscreteLowRankGaussianProcess[D, DDomain, Value](domain, m, lambdas, U)
  }

  def apply[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
    mean: DiscreteField[D, DDomain, Value],
    klBasis: KLBasis[D, DDomain, Value]
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {

    for (Eigenpair(_, phi) <- klBasis) {
      require(phi.domain == mean.domain)
    }

    val domain = mean.domain
    val meanVec: DenseVector[Double] = DiscreteField.vectorize(mean)
    val varianceVec = DenseVector.zeros[Double](klBasis.size)
    val basisMat = DenseMatrix.zeros[Double](meanVec.length, klBasis.size)
    for ((eigenPair, i) <- klBasis.zipWithIndex) yield {
      val Eigenpair(lambda, phi) = eigenPair
      basisMat(::, i) := DiscreteField.vectorize[D, DDomain, Value](phi: DiscreteField[D, DDomain, Value])(vectorizer)
      varianceVec(i) = lambda
    }
    new DiscreteLowRankGaussianProcess[D, DDomain, Value](domain, meanVec, varianceVec, basisMat)
  }

  def unapply[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
    dgp: DiscreteLowRankGaussianProcess[D, DDomain, Value]
  ): Option[(DiscreteDomain[D], DenseVector[Double], DenseMatrix[Double], DenseVector[Double])] = {
    Option(dgp.domain, dgp.meanVector, dgp.basisMatrix, dgp.variance)
  }

  /**
   * Discrete implementation of [[LowRankGaussianProcess.regression]]
   */
  def regression[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
    gp: DiscreteLowRankGaussianProcess[D, DDomain, Value],
    trainingData: IndexedSeq[(PointId, Value, MultivariateNormalDistribution)],
    naNStrategy: NaNStrategy = NanIsNumericValue
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {

    val LowRankRegressionComputation(_Minv, yVec, mVec, _QtL) =
      LowRankRegressionComputation.fromDiscreteLowRankGP(gp, trainingData, naNStrategy)
    val mean_coeffs = (_Minv * _QtL) * (yVec - mVec)

    // val mean_p = gp.instance(mean_coeffs)
    val mean_pVector = gp.instanceVector(mean_coeffs)

    val D = breeze.linalg.diag(DenseVector(gp.variance.map(math.sqrt).toArray))
    val Sigma = D * _Minv * D
    val SVD(innerU, innerD2, _) = breeze.linalg.svd(Sigma)

    val lambdas_p = DenseVector[Double](innerD2.toArray)

    // we do the following computation
    // val eigenMatrix_p = gp.eigenMatrix * innerU // IS this correct?
    // but in parallel
    val eigenMatrix_p = DenseMatrix.zeros[Double](gp.basisMatrix.rows, innerU.cols)
    for (rowInd <- ParVector.range(0, gp.basisMatrix.rows)) {

      // TODO maybe this strange transposing can be alleviated? It seems breeze does not support
      // row-vector matrix multiplication
      eigenMatrix_p(rowInd, ::) := (innerU.t * gp.basisMatrix(rowInd, ::).t).t
    }

    new DiscreteLowRankGaussianProcess(gp.domain, mean_pVector, lambdas_p, eigenMatrix_p)
  }

  /**
   * Discrete implementation of [[LowRankGaussianProcess.regressionMean]]
   */
  def regressionMean[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
    gp: DiscreteLowRankGaussianProcess[D, DDomain, Value],
    trainingData: IndexedSeq[(PointId, Value, MultivariateNormalDistribution)],
    naNStrategy: NaNStrategy = NanIsNumericValue
  )(implicit vectorizer: Vectorizer[Value]): DiscreteField[D, DDomain, Value] = {

    val LowRankRegressionComputation(_Minv, yVec, mVec, _QtL) =
      LowRankRegressionComputation.fromDiscreteLowRankGP(gp, trainingData, naNStrategy)
    val mean_coeffs = (_Minv * _QtL) * (yVec - mVec)
    gp.instance(mean_coeffs)
  }

  def createUsingPCA[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
    dc: DataCollection[D, DDomain, Value],
    stoppingCriterion: PivotedCholesky.StoppingCriterion = PivotedCholesky.RelativeTolerance(0)
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {
    if (dc.size < 3) {
      throw new IllegalArgumentException(s"The datacollection contains only ${dc.size} items. At least 3 are needed")
    }

    val fields = dc.fields(NearestNeighborInterpolator())
    createUsingPCA(dc.reference, fields, stoppingCriterion)
  }

  /**
   * Creates a new DiscreteLowRankGaussianProcess, where the mean and covariance matrix are estimated from the given
   * sample of continuous vector fields using Principal Component Analysis.
   */
  def createUsingPCA[D: NDSpace, DDomain[D] <: DiscreteDomain[D], Value](
    domain: DDomain[D],
    fields: Seq[Field[D, Value]],
    stoppingCriterion: PivotedCholesky.StoppingCriterion
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {

    val dim = vectorizer.dim

    val n = fields.size
    val p = domain.pointSet.numberOfPoints

    // create the data matrix - note, it will be manipulated inplace
    val X = DenseMatrix.zeros[Double](n, p * dim)
    for (p1 <- new ParVector(fields.zipWithIndex.toVector); p2 <- domain.pointSet.pointsWithId) {
      val (f, i) = p1
      val (x, ptId) = p2
      val ux = vectorizer.vectorize(f(x))
      X(i, ptId.id * dim until (ptId.id + 1) * dim) := ux.t
    }

    // demean the data matrix
    val m: DenseVector[Double] = breeze.stats.mean(X(::, breeze.linalg.*)).inner
    for (i <- 0 until X.rows) {
      X(i, ::) := X(i, ::) - m.t
    }

    // Whether it is more efficient to compute the PCA using the gram matrix or the covariance matrix, depends on
    // whether we have more examples than variables.
    val (basisMat, varianceVector) = if (X.rows > X.cols) {
      val (u, d2) = PivotedCholesky.computeApproximateEig(X.t * X * (1.0 / (n - 1)), stoppingCriterion)
      (u, d2)
    } else {

      val (v, d2) = PivotedCholesky.computeApproximateEig(X * X.t * (1.0 / (n - 1)), stoppingCriterion)

      val D = d2.map(v => Math.sqrt(v))
      val Dinv = D.map(d => if (d > 1e-6) 1.0 / d else 0.0)

      // The following is an efficient version to compute vt.t * diag(DInv) / Math.sqrt(n - 1)
      for (i <- 0 until Dinv.length) {
        v(::, i) := v(::, i) * Dinv(i) / Math.sqrt(n - 1)
      }

      // The final basis matrix is commputed based on the data matrix and v
      val U: DenseMatrix[Double] = X.t * v
      (U, d2)

    }

    new DiscreteLowRankGaussianProcess(domain, m, varianceVector, basisMat)
  }

  private def basisMatrixToCov[D: NDSpace, Value](domain: DiscreteDomain[D],
                                                  variance: DenseVector[Double],
                                                  basisMatrix: DenseMatrix[Double]
  ) = {

    val outputDim = basisMatrix.rows / domain.pointSet.numberOfPoints
    def cov(ptId1: PointId, ptId2: PointId): DenseMatrix[Double] = {

      // same as commented line above, but just much more efficient (as breeze does not have diag matrix,
      // the upper command does a lot of  unnecessary computations
      val covValue = DenseMatrix.zeros[Double](outputDim, outputDim)

      for (i <- ParVector.range(0, outputDim)) {
        val ind1 = ptId1.id * outputDim + i
        var j = 0
        while (j < outputDim) {
          val ind2 = ptId2.id * outputDim + j
          var k = 0
          var valueIJ = 0.0
          while (k < basisMatrix.cols) {
            valueIJ += basisMatrix(ind1, k) * basisMatrix(ind2, k) * variance(k)
            k += 1
          }
          covValue(i, j) = valueIJ
          j += 1
        }
      }

      covValue
    }

    DiscreteMatrixValuedPDKernel(domain, cov, outputDim)
  }

  def realignment[D: NDSpace, DDomain[DD] <: DiscreteDomain[DD], Value](
    model: DiscreteLowRankGaussianProcess[D, DDomain, Value],
    ids: IndexedSeq[PointId],
    withRotation: Boolean,
    diagonalize: Boolean
  )(implicit vectorizer: Vectorizer[Value], realignmentType: Realignment[D, DDomain, Value]): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {
    realignmentType.realign(model, ids, withRotation, diagonalize)
  }

  private[scalismo] def realignmentComputation(mean: DenseVector[Double],
                                     basis: DenseMatrix[Double],
                                     s: DenseVector[Double],
                                     p: DenseMatrix[Double],
                                     ids: IndexedSeq[Int],
                                     dim: Int,
                                     diagonalize: Boolean,
                                     projectMean: Boolean
  ): (DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
    val x = for // prepare indices
      id <- ids
      d <- 0 until dim
    yield id * dim + d
    // prepare the majority of the projection matrix
    val px = p(x, ::).toDenseMatrix
    val ptpipt = breeze.linalg.pinv(px.t * px) * px.t

    // performs the actual projection. batches all basis vectors
    // p -> projection rank, n number of indexes*dim, r cols of basis, N rows of basis
    val alignedC = ptpipt * basis(x, ::).toDenseMatrix // pxn * nxr
    val alignedEigf = basis - p * alignedC // Nxr - Nxp * pxr
    val alignedMean = if projectMean then // if desired projects the mean vector as well
      val alignedMc = ptpipt * mean // same projection with r==1
      mean - p * alignedMc
    else mean

    // rediagonalize. You can skip this if you ONLY sample from the resulting model
    val (newbasis, news) =
      if diagonalize then GramDiagonalize.rediagonalizeGram(alignedEigf, s)
      else (alignedEigf, s)

    (alignedMean, newbasis, news)
  }

}

// Explicit variants for 1D, 2D and 3D
object DiscreteLowRankGraussianProcess1D {

  /**
   * Creates a new DiscreteLowRankGaussianProcess by discretizing the given gaussian process at the domain points.
   */
  def apply[DDomain[D] <: DiscreteDomain[D], Value](domain: DDomain[_1D], gp: LowRankGaussianProcess[_1D, Value])(
    implicit vectorizer: Vectorizer[Value]
  ): DiscreteLowRankGaussianProcess[_1D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.apply[_1D, DDomain, Value](domain, gp)
  }

  /**
   * Creates a new DiscreteLowRankGaussianProcess from the given mean field and klBasis
   */
  def apply[DDomain[D] <: DiscreteDomain[D], Value](
    mean: DiscreteField[_1D, DDomain, Value],
    klBasis: KLBasis[_1D, DDomain, Value]
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[_1D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.apply[_1D, DDomain, Value](mean, klBasis)
  }
}

object DiscreteLowRankGraussianProcess2D {

  /**
   * Creates a new DiscreteLowRankGaussianProcess by discretizing the given gaussian process at the domain points.
   */
  def apply[DDomain[D] <: DiscreteDomain[D], Value](domain: DDomain[_2D], gp: LowRankGaussianProcess[_2D, Value])(
    implicit vectorizer: Vectorizer[Value]
  ): DiscreteLowRankGaussianProcess[_2D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.apply[_2D, DDomain, Value](domain, gp)
  }

  /**
   * Creates a new DiscreteLowRankGaussianProcess from the given mean field and klBasis
   */
  def apply[DDomain[D] <: DiscreteDomain[D], Value](
    mean: DiscreteField[_2D, DDomain, Value],
    klBasis: KLBasis[_2D, DDomain, Value]
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[_2D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.apply[_2D, DDomain, Value](mean, klBasis)
  }
}

object DiscreteLowRankGraussianProcess3D {

  /**
   * Creates a new DiscreteLowRankGaussianProcess by discretizing the given gaussian process at the domain points.
   */
  def apply[DDomain[D] <: DiscreteDomain[D], Value](domain: DDomain[_3D], gp: LowRankGaussianProcess[_3D, Value])(
    implicit vectorizer: Vectorizer[Value]
  ): DiscreteLowRankGaussianProcess[_3D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.apply[_3D, DDomain, Value](domain, gp)
  }

  /**
   * Creates a new DiscreteLowRankGaussianProcess from the given mean field and klBasis
   */
  def apply[DDomain[D] <: DiscreteDomain[D], Value](
    mean: DiscreteField[_3D, DDomain, Value],
    klBasis: KLBasis[_3D, DDomain, Value]
  )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[_3D, DDomain, Value] = {
    DiscreteLowRankGaussianProcess.apply[_3D, DDomain, Value](mean, klBasis)
  }
}


trait Realignment[D: NDSpace, DDomain[DD] <: DiscreteDomain[DD], Value] {
  def realign(
               model: DiscreteLowRankGaussianProcess[D, DDomain, Value],
               ids: IndexedSeq[PointId],
               withRotation: Boolean,
               diagonalize: Boolean
             )(implicit vectorizer: Vectorizer[Value]): DiscreteLowRankGaussianProcess[D, DDomain, Value] = {
    val d = model.domain.pointSet.points.next().dimensionality  // TODO shorter expression
    // build the projection matrix for the desired pose
    val p = {
      val pt = breeze.linalg.tile(DenseMatrix.eye[Double](d), model.domain.pointSet.numberOfPoints, 1)
      if withRotation then
        val center = ids.map(id => model.domain.pointSet.point(id).toVector).reduce(_ + _).map(_ / ids.length).toPoint
        val pr = getTangentialSpeedMatrix(model.domain, center)
        DenseMatrix.horzcat(pt, pr)
      else pt
    }
    // call the realignment implementation
    val (nmean, nbasis, nvar) = realignmentComputation(model.meanVector,
                                  model.basisMatrix,
                                  model.variance,
                                  p,
                                  ids.map(_.id),
                                  dim = d,
                                  diagonalize = diagonalize,
                                  projectMean = false)

    new DiscreteLowRankGaussianProcess[D, DDomain, Value](model.domain, nmean, nvar, nbasis)
  }

  def centeredP(domain: DiscreteDomain[D], center: Point[D]): DenseMatrix[Double] = {
    val dim = center.dimensionality
    val np = domain.pointSet.numberOfPoints
    require(dim >= 2, "requires at least two dimensions to calculate a tangential speed")
    // build centered data matrix
    val x = DenseMatrix.zeros[Double](dim, np)
    val c = center.toBreezeVector
    for (p, i) <- domain.pointSet.points.zipWithIndex do x(::, i) := p.toBreezeVector - c
    x
  }

  def getTangentialSpeedMatrix(domain: DDomain[D], center: Point[D]): DenseMatrix[Double]

}

given givenRealignment3D: Realignment[_3D, DiscreteDomain, EuclideanVector[_3D]] with
  def getTangentialSpeedMatrix(domain: DiscreteDomain[_3D], center: Point[_3D]): DenseMatrix[Double] = {
    val dim = center.dimensionality
    val np = domain.pointSet.numberOfPoints
    val x = centeredP(domain, center)

    val pr = DenseMatrix.zeros[Double](np * dim, 3)
    // the derivative of the rotation matrices
    val dr = new DenseMatrix[Double](9, 3,
      Array(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )
    // get tangential speed
    val dx = dr * x
    for i <- 0 until 3 do
      val v = dx(3 * i until 3 * i + 3, ::).toDenseVector
      pr(::, i) := v / breeze.linalg.norm(v)
    pr
  }

given givenRealignment2D: Realignment[_2D, DiscreteDomain, EuclideanVector[_2D]] with
  def getTangentialSpeedMatrix(domain: DiscreteDomain[_2D], center: Point[_2D]): DenseMatrix[Double] = {
    val dim = center.dimensionality
    val np = domain.pointSet.numberOfPoints
    val x = centeredP(domain, center)

    //derivative of the rotation matrix
    val dr = new DenseMatrix[Double](2, 2, Array(0.0, -1.0, 1.0, 0.0))
    val dx = (dr * x).reshape(dim * np, 1)
    val n = breeze.linalg.norm(dx, breeze.linalg.Axis._0)
    dx / n(0)
  }

implicit object Realignment3D extends Realignment[_3D, DiscreteDomain, EuclideanVector[_3D]] {
  override def getTangentialSpeedMatrix(domain: DiscreteDomain[_3D], center: Point[_3D]): DenseMatrix[Double] = {
    val dim = center.dimensionality
    val np = domain.pointSet.numberOfPoints
    val x = centeredP(domain, center)

    val pr = DenseMatrix.zeros[Double](np * dim, 3)
    // the derivative of the rotation matrices
    val dr = new DenseMatrix[Double](9, 3,
      Array(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )
    // get tangential speed
    val dx = dr * x
    for i <- 0 until 3 do
      val v = dx(3 * i until 3 * i + 3, ::).toDenseVector
      pr(::, i) := v / breeze.linalg.norm(v)
    pr
  }
//  def realign(
//               model: DiscreteLowRankGaussianProcess[_3D, DiscreteDomain, EuclideanVector[_3D]],
//               ids: IndexedSeq[PointId],
//               withRotation: Boolean,
//               diagonalize: Boolean
//             )(implicit vectorizer: Vectorizer[EuclideanVector[_3D]]): DiscreteLowRankGaussianProcess[_3D, DiscreteDomain, EuclideanVector[_3D]] = {
//    // build the projection matrix for the desired pose
//    val p = {
//      val pt = breeze.linalg.tile(DenseMatrix.eye[Double](3), model.domain.pointSet.numberOfPoints, 1)
//      if withRotation then
//        val center = ids.map(id => model.domain.pointSet.point(id).toVector).reduce(_ + _).map(_ / ids.length).toPoint
//        val pr = getTangentialSpeedMatrix(model.domain, center)
//        DenseMatrix.horzcat(pt, pr)
//      else pt
//    }
//    // call the realignment implementation
//    val (nmean, nbasis, nvar) = realignmentComputation(model.meanVector,
//      model.basisMatrix,
//      model.variance,
//      p,
//      ids.map(_.id),
//      dim = 3,
//      diagonalize = diagonalize,
//      projectMean = false
//    )
//
//    new DiscreteLowRankGaussianProcess[_3D, DiscreteDomain, EuclideanVector[_3D]](model.domain, nmean, nvar, nbasis)
//  }


}


implicit object Realignment2D extends Realignment[_2D, DiscreteDomain, EuclideanVector[_2D]] {

  override def getTangentialSpeedMatrix(domain: DiscreteDomain[_2D], center: Point[_2D]): DenseMatrix[Double] = {
    val dim = center.dimensionality
    val np = domain.pointSet.numberOfPoints
    val x = centeredP(domain, center)

    //derivative of the rotation matrix
    val dr = new DenseMatrix[Double](2, 2, Array(0.0, -1.0, 1.0, 0.0))
    val dx = (dr * x).reshape(dim * np, 1)
    val n = breeze.linalg.norm(dx, breeze.linalg.Axis._0)
    dx / n(0)
  }
//  def realign(
//               model: DiscreteLowRankGaussianProcess[_2D, DiscreteDomain, EuclideanVector[_2D]],
//               ids: IndexedSeq[PointId],
//               withRotation: Boolean,
//               diagonalize: Boolean
//             )(implicit vectorizer: Vectorizer[EuclideanVector[_2D]]): DiscreteLowRankGaussianProcess[_2D, DiscreteDomain, EuclideanVector[_2D]] = {
//    // build the projection matrix for the desired pose
//    val p = {
//      val pt = breeze.linalg.tile(DenseMatrix.eye[Double](3), model.domain.pointSet.numberOfPoints, 1)
//      if withRotation then
//        val center = ids.map(id => model.domain.pointSet.point(id).toVector).reduce(_ + _).map(_ / ids.length).toPoint
//        val pr = getTangentialSpeedMatrix(model.domain, center)
//        DenseMatrix.horzcat(pt, pr)
//      else pt
//    }
//    // call the realignment implementation
//    val (nmean, nbasis, nvar) = realignmentComputation(model.meanVector,
//      model.basisMatrix,
//      model.variance,
//      p,
//      ids.map(_.id),
//      dim = 3,
//      diagonalize = diagonalize,
//      projectMean = false
//    )
//
//    new DiscreteLowRankGaussianProcess[_3D, DiscreteDomain, EuclideanVector[_3D]](model.domain, nmean, nvar, nbasis)
//  }
}


extension(model: DiscreteLowRankGaussianProcess[_3D, DiscreteDomain, EuclideanVector[_3D]])
  def realign2D(ids: IndexedSeq[PointId], withRotation: Boolean = true, diagonalize: Boolean = true): DiscreteLowRankGaussianProcess[_3D, DiscreteDomain, EuclideanVector[_3D]] = {
    Realignment3D.realign(model, ids, withRotation, diagonalize)
  }

extension(model: DiscreteLowRankGaussianProcess[_2D, DiscreteDomain, EuclideanVector[_2D]])
  def realign3D(ids: IndexedSeq[PointId], withRotation: Boolean = true, diagonalize: Boolean = true): DiscreteLowRankGaussianProcess[_2D, DiscreteDomain, EuclideanVector[_2D]] = {
    Realignment2D.realign(model, ids, withRotation, diagonalize)
  }




