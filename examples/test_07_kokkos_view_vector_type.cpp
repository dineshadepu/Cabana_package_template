/*
  How to run this examples:
  ./examples/04ObliqueParticleWallDifferentAngles ../examples/inputs/04_oblique_particle_wall_different_angles.json ./

*/
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaNewPkg.hpp>
#include <Eigen/Dense>

using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = typename exec_space::memory_space;
using ExecutionSpace = exec_space;
using MemorySpace = memory_space;

using vec_double_type = Cabana::MemberTypes<double[3]>;
using matrix_double_type = Cabana::MemberTypes<double[9]>;

using aosoa_vec_double_type = Cabana::AoSoA<vec_double_type, memory_space, 1>;
using aosoa_mat_double_type = Cabana::AoSoA<matrix_double_type, memory_space, 1>;



typedef Kokkos::View<double*>   ViewVectorType;
typedef Kokkos::View<double**>  ViewMatrixType;

void compute_eigen_values(auto no_elements, auto moi, auto moi_principal, auto rot_mat){
  Eigen::Matrix3d I;
  for ( std::size_t i = 0; i < no_elements; ++i )
    {
      // create a moi matrix of particle i so that we can call eigen
      // function to computer the eigen values and
      I << moi( i, 0 ), moi( i, 1 ), moi( i, 2 ),
        moi( i, 3 ), moi( i, 4 ), moi( i, 5 ),
        moi( i, 6 ), moi( i, 7 ), moi( i, 8 );
      Eigen::EigenSolver<Eigen::Matrix3d> es(I);

      moi_principal( i, 0 ) = es.eigenvalues()[0].real();
      moi_principal( i, 1 ) = es.eigenvalues()[1].real();
      moi_principal( i, 2 ) = es.eigenvalues()[2].real();

      rot_mat( i, 0 ) = es.eigenvectors().col(0).real()[0];
      rot_mat( i, 3 ) = es.eigenvectors().col(0).real()[1];
      rot_mat( i, 6 ) = es.eigenvectors().col(0).real()[2];

      rot_mat( i, 1 ) = es.eigenvectors().col(1).real()[0];
      rot_mat( i, 4 ) = es.eigenvectors().col(1).real()[1];
      rot_mat( i, 7 ) = es.eigenvectors().col(1).real()[2];

      rot_mat( i, 2 ) = es.eigenvectors().col(2).real()[0];
      rot_mat( i, 5 ) = es.eigenvectors().col(2).real()[1];
      rot_mat( i, 8 ) = es.eigenvectors().col(2).real()[2];
    }
}

void view_vector_testing(){

  // We first copy the moment of inertia
  // This is on the device
  aosoa_mat_double_type moi;
  aosoa_vec_double_type moi_principal;
  aosoa_mat_double_type rot_mat;
  moi.resize(3);
  moi_principal.resize(3);
  rot_mat.resize(3);

  auto moi_slice = Cabana::slice<0>( moi,    "moi");
  auto moi_principal_slice = Cabana::slice<0>( moi_principal,    "moi_principal");
  auto rot_mat_slice = Cabana::slice<0>( rot_mat,    "rot_mat");

  auto initialize_moi_func = KOKKOS_LAMBDA( const int i )
    {
      for (int j=0; j < 9; j++){
        moi_slice( i, j ) = i * j + j;
      }
    };
  Kokkos::RangePolicy<exec_space> policy( 0, moi.size() );
  Kokkos::parallel_for( "moi_initialize", policy,
                        initialize_moi_func );

  ViewMatrixType moi_principal_tmp_device( "moi_principal_tmp_device", 3, 3);
  ViewMatrixType moi_tmp_device( "moi_tmp_device", 3, 9);
  ViewMatrixType rot_mat_tmp_device( "rot_mat_tmp_device", 3, 9);

  auto copy_to_local_moi_func = KOKKOS_LAMBDA( const int i )
    {
      for (int j=0; j < 9; j++){
        moi_tmp_device( i, j ) = moi_slice( i, j );
        rot_mat_tmp_device( i, j ) = 0.;
      }
      moi_principal_tmp_device( i, 0 ) = 0.;
      moi_principal_tmp_device( i, 1 ) = 0.;
      moi_principal_tmp_device( i, 2 ) = 0.;
    };
  Kokkos::parallel_for( "moi_copy", policy,
                        copy_to_local_moi_func );

  ViewMatrixType::HostMirror moi_principal_tmp_host = Kokkos::create_mirror_view( moi_principal_tmp_device  );
  ViewMatrixType::HostMirror moi_tmp_host = Kokkos::create_mirror_view( moi_tmp_device );
  ViewMatrixType::HostMirror rot_mat_tmp_host = Kokkos::create_mirror_view( rot_mat_tmp_device );

  // compute the eigen values and eigen vectors of the moi
  compute_eigen_values( 3, moi_tmp_host, moi_principal_tmp_host, rot_mat_tmp_host );

  // copy back to the device
  Kokkos::deep_copy( moi_principal_tmp_device, moi_principal_tmp_host );
  Kokkos::deep_copy( moi_tmp_device, moi_tmp_host );
  Kokkos::deep_copy( rot_mat_tmp_device, rot_mat_tmp_host );

  auto copy_to_main_slice_func = KOKKOS_LAMBDA( const int i )
    {
      for (int j=0; j < 9; j++){
        rot_mat_slice( i, j ) = rot_mat_tmp_device( i, j );
      }
      moi_principal_slice( i, 0 ) = moi_principal_tmp_device( i, 0 );
      moi_principal_slice( i, 1 ) = moi_principal_tmp_device( i, 1 );
      moi_principal_slice( i, 2 ) = moi_principal_tmp_device( i, 2 );
    };
  Kokkos::parallel_for( "moi_copy_to_slice", policy,
                        copy_to_main_slice_func );

  // for ( std::size_t i = 0; i < moi.size(); ++i )
  //   {
  //     std::cout << "Rotation matrix of " << i << " is:"
  //               << std::endl;
  //     for ( std::size_t j = 0; j < 9; ++j )
  //       {
  //         std::cout <<  moi_slice(i, j) << ", " ;
  //       }
  //     std::cout << std::endl;

  //     std::cout << "MOI view matrix device " << i << " is:"
  //               << std::endl;
  //     for ( std::size_t j = 0; j < 9; ++j )
  //       {
  //         std::cout <<  moi_tmp_device(i, j) << ", " ;
  //       }
  //     std::cout << std::endl;

  //     std::cout << "HOST MOI view matrix host " << i << " is:"
  //               << std::endl;
  //     for ( std::size_t j = 0; j < 9; ++j )
  //       {
  //         std::cout <<  moi_tmp_host(i, j) << ", " ;
  //       }
  //     std::cout << std::endl;
  //   }
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );

  view_vector_testing();

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
