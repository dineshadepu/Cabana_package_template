/*
  How to run this examples:
  ./examples/04ObliqueParticleWallDifferentAngles ../examples/inputs/04_oblique_particle_wall_different_angles.json ./

*/
#include <iostream>

#include <CabanaNewPkg.hpp>
#include <Eigen/Dense>



int main( int argc, char* argv[] )
{
  Eigen::Matrix3d I;
  I << 1., 2., 3.,
    33., 11., 232.,
    3., 53., 12.;

  Eigen::EigenSolver<Eigen::Matrix3d> es(I);

  std::cout << "The eigenvalues of I are:\n" << es.eigenvalues() << std::endl;
  std::cout << "The eigenvectors of I are:\n" << es.eigenvectors().col(0).real() << std::endl;

  return 0;
}
