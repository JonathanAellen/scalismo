package scalismo.statisticalmodel

import breeze.linalg.DenseMatrix
import scalismo.common.DiscreteDomain
import scalismo.geometry.*

/**
 * types the whole discrete low rank gp to make sure that it is applied to the appropriate models. The value type could
 * be left out if the user knows what to do.
 */
trait RealignExtendedBasis[D: NDSpace, Value]:

  def useTranslation: Boolean
  def getBasis[DDomain[DD] <: DiscreteDomain[DD]](model: DiscreteLowRankGaussianProcess[D, DDomain, Value],
                                                  center: Point[D]
  ): DenseMatrix[Double]
  def centeredP[DDomain[DD] <: DiscreteDomain[DD]](domain: DDomain[D], center: Point[D]): DenseMatrix[Double] = {
    // build centered data matrix
    val x = DenseMatrix.zeros[Double](center.dimensionality, domain.pointSet.numberOfPoints)
    val c = center.toBreezeVector
    for (p, i) <- domain.pointSet.points.zipWithIndex do x(::, i) := p.toBreezeVector - c
    x
  }

object RealignExtendedBasis:
  /**
   * returns a projection basis for rotation. that is the tangential speed for the rotations around the three cardinal
   * directions.
   */
  given realignBasis3D: RealignExtendedBasis[_3D, EuclideanVector[_3D]] with
    def useTranslation: Boolean = true
    def getBasis[DDomain[DD] <: DiscreteDomain[DD]](
      model: DiscreteLowRankGaussianProcess[_3D, DDomain, EuclideanVector[_3D]],
      center: Point[_3D]
    ): DenseMatrix[Double] = {
      val np = model.domain.pointSet.numberOfPoints
      val x = centeredP(model.domain, center)

      val pr = DenseMatrix.zeros[Double](np * 3, 3)
      // the derivative of the rotation matrices
      val dr = new DenseMatrix[Double](9,
                                       3,
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

  /**
   * returns a projection basis for rotation. that is the tangential speed for the single 2d rotation.
   */
  given realignBasis2D: RealignExtendedBasis[_2D, EuclideanVector[_2D]] with
    def useTranslation: Boolean = true
    def getBasis[DDomain[DD] <: DiscreteDomain[DD]](
      model: DiscreteLowRankGaussianProcess[_2D, DDomain, EuclideanVector[_2D]],
      center: Point[_2D]
    ): DenseMatrix[Double] = {
      val np = model.domain.pointSet.numberOfPoints
      val x = centeredP(model.domain, center)

      // derivative of the rotation matrix
      val dr = new DenseMatrix[Double](2, 2, Array(0.0, -1.0, 1.0, 0.0))
      val dx = (dr * x).reshape(2 * np, 1)
      val n = breeze.linalg.norm(dx, breeze.linalg.Axis._0)
      dx / n(0)
    }
