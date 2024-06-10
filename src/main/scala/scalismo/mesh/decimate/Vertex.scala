package scalismo.mesh.decimate

import scalismo.geometry.{_3D, Point}

case class Vertex(
  var point: Point[_3D],
  border: Boolean,
  var tstart: Int = 0,
  var tcount: Int = 0,
  var q: SymmetricMatrix = SymmetricMatrix(0.0)
)