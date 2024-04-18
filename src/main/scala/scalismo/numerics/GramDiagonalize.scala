package scalismo.numerics

import breeze.linalg._
import breeze.linalg.given 

object GramDiagonalize {

  /**
   * Given a non orthogonal basis nxr and the variance (squared [eigen]scalars) of that basis,
   * returns an orthonormal basis with the adjusted variance.
   * sets small eigenvalues to zero.
   */
  def rediagonalizeGram(basis: DenseMatrix[Double], s: DenseVector[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
    //val l: DenseMatrix[Double] = basis(*, ::) * breeze.numerics.sqrt(s)
    val l: DenseMatrix[Double] = DenseMatrix.zeros[Double](basis.rows, basis.cols)
    val sqs: DenseVector[Double] = breeze.numerics.sqrt(s)
    for i <- 0 until basis.cols do l(::,i) :=  sqs(i) * basis(::,i)

    val gram = l.t * l
    val svd = breeze.linalg.svd(gram)
    val newS: DenseVector[Double] = breeze.numerics.sqrt(svd.S).map(d => if(d > 1e-10) 1.0/d else 0.0)

    //val newbasis: DenseMatrix[Double] = l * (svd.U(*, ::) * newS)
    val newbasis: DenseMatrix[Double] = DenseMatrix.zeros[Double](basis.rows, basis.cols)
    for i <- 0 until basis.cols do newbasis(::,i) := svd.U(::,i) * newS(i)
    newbasis := l * newbasis

    (newbasis, svd.S)
  }

}
