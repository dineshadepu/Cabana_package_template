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


template<typename Y>
auto create_points(
                   double x_start, double x_end, int x_steps,
                   double y_start, double y_end, int y_steps,
                   Y & particles) {

  int size = x_steps * y_steps;
  particles->resize(size);
  auto x = particles->slicePosition();
  auto u = particles->sliceVelocity();
  auto m = particles->sliceMass();
  auto rad = particles->sliceRadius();
  auto k = particles->sliceRadius();

  // Calculate the number of rows and columns based on length, height, and spacing
  double x_step_size = (x_end - x_start) / (x_steps);
  double y_step_size = (y_end - y_start) / (y_steps);

  int global_index = 0;
  for (int i = 0; i < x_steps; ++i) {
    for (int j = 0; j < y_steps; ++j) {
      x( global_index, 0 ) = x_start + i * x_step_size;
      x( global_index, 1 ) = y_start + j * y_step_size;
      x( global_index, 2 ) = 0.;
      m( global_index ) = 1.;
      rad( global_index ) = 0.;
      global_index += 1;
    }
  }
}


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
  // get the total number of particles
  auto particles = std::make_shared<
    CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1);


  double x_start = -5.;
  double x_end = 6.;
  double y_start = -5.;
  double y_end = 6.;
  create_points(x_start, x_end, 11,
                y_start, y_end, 11,
                particles);

  // ====================================================
  //            Custom particle initialization
  // ====================================================
  double neighborhood_radius = 1.2;
  double grid_min[3] = { x_start - neighborhood_radius , y_start - neighborhood_radius , -2. * neighborhood_radius };
  double grid_max[3] = { x_end + neighborhood_radius , y_end + neighborhood_radius , 2. * neighborhood_radius };

  double cell_ratio = 1.0;
  using ListAlgorithm = Cabana::FullNeighborTag;
  using ListType = Cabana::VerletList<memory_space, ListAlgorithm,
                                      Cabana::VerletLayoutCSR,
                                      Cabana::TeamOpTag>;

  auto positions = particles->slicePosition();
  auto verlet_list = std::make_shared<ListType>( positions, 0, positions.size(), neighborhood_radius,
                        cell_ratio, grid_min, grid_max );

  std::cout << "=================== " << std::endl;
  std::cout << "=================== " << std::endl;
  std::cout << " before the nested loop equation " << std::endl;
  std::cout << "=================== " << std::endl;
  std::cout << "=================== " << std::endl;

  // Equation testNestedLoopEquation wrote here manually
  auto x = particles->slicePosition();
  auto rad = particles->sliceRadius();

  auto force_full = KOKKOS_LAMBDA( const int i, const int j )
    {
      /*
        Common to all equations in SPH.

        We compute:
        1.the vector passing from j to i
        2. Distance between the points i and j
        3. Distance square between the points i and j
        4. Velocity vector difference between i and j
        5. Kernel value
        6. Derivative of kernel value
      */
      double pos_i[3] = {x( i, 0 ),
                         x( i, 1 ),
                         x( i, 2 )};

      double pos_j[3] = {x( j, 0 ),
                         x( j, 1 ),
                         x( j, 2 )};

      double pos_ij[3] = {x( i, 0 ) - x( j, 0 ),
                          x( i, 1 ) - x( j, 1 ),
                          x( i, 2 ) - x( j, 2 )};

      // squared distance
      double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
      // distance between i and j
      double rij = sqrt(r2ij);
      /*
        ====================================
        End: common to all equations in SPH.
        ====================================
      */
      // find the force if the particles are overlapping

      if (rij < neighborhood_radius) {
        rad( i ) += 1.;
      }
    };

  Kokkos::RangePolicy<Kokkos::OpenMP> policy(x.size(), 0);

  Cabana::neighbor_parallel_for( policy,
                                 force_full,
                                 *verlet_list,
                                 Cabana::FirstNeighborsTag(),
                                 Cabana::SerialOpTag(),
                                 "CabanaDEM::ForceFull" );
  Kokkos::fence();

  std::cout << "=================== " << std::endl;
  std::cout << "=================== " << std::endl;
  std::cout << " after the nested loop equation " << std::endl;
  std::cout << "=================== " << std::endl;
  std::cout << "=================== " << std::endl;

  // Set output folder
  // particles->set_output_folder("test_compare_python_01_nested_for_loop_output");
  // particles->set_output_folder("./");
  particles->output(0, 0.);
  std::cout << "=================== " << std::endl;
  std::cout << "=================== " << std::endl;
  std::cout << " after the output " << std::endl;
  std::cout << "=================== " << std::endl;
  std::cout << "=================== " << std::endl;

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
