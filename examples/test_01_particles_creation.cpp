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

#define DIM 3


// Simulate two spherical particles colliding head on
double CreateParticles()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana create particles example\n" << std::endl;


  // ====================================================
  //             Use default Kokkos spaces
  // ====================================================
  // using exec_space = Kokkos::DefaultExecutionSpace;
  // using memory_space = typename exec_space::memory_space;
  using exec_space = Kokkos::OpenMP;
  using memory_space = typename exec_space::memory_space;


  // ====================================================
  //                 Particle generation
  // ====================================================
  auto particles = std::make_shared<
    CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1);

  // ====================================================
  //            Custom particle initialization
  // ====================================================
  auto x_p = particles->slicePosition();
  auto u_p = particles->sliceVelocity();
  auto m_p = particles->sliceMass();
  auto rad_p = particles->sliceRadius();
  auto k_p = particles->sliceRadius();

  double radius_p_inp = 0.1;
  double k_p_inp = 1e5;

  auto particles_init_functor = KOKKOS_LAMBDA( const int pid )
    {
      // Initial conditions: displacements and velocities
      double m_p_i = 4. / 3. * M_PI * radius_p_inp * radius_p_inp * radius_p_inp * 1000.;
      double I_p_i = 2. / 5. * m_p_i * radius_p_inp * radius_p_inp;
      x_p( pid, 0 ) = 0.;
      x_p( pid, 1 ) = radius_p_inp + radius_p_inp / 1000000.;
      x_p( pid, 2 ) = 0.;
      u_p( pid, 0 ) = 0.;
      u_p( pid, 1 ) = 0.;
      u_p( pid, 2 ) = 0.0;

      m_p( pid ) = m_p_i;
      k_p( pid ) = k_p_inp;
      rad_p( pid ) = radius_p_inp;
    };
  particles->updateParticles( exec_space{}, particles_init_functor );

  particles->resize(20);

  auto x_p_20 = particles->slicePosition();
  auto u_p_20 = particles->sliceVelocity();

  for (int i=0; i < u_p_20.size(); i++){
    x_p_20( i, 0 ) = 0.;
    x_p_20( i, 1 ) = 3 * i * radius_p_inp + radius_p_inp / 1000000.;
    x_p_20( i, 2 ) = 0.;

    u_p_20( i, 0 ) = 0.;
    u_p_20( i, 1 ) = 1.;
    u_p_20( i, 2 ) = 0.;
  }

  return 0;

}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );
  CreateParticles();

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
