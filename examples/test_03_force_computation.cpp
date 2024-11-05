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
void create_points(Y & particles) {
    int num_points_per_row = 11;
    double step = 1.0 / (num_points_per_row - 1);

    std::vector<std::pair<double, double>> points;

    for (int i = 0; i < num_points_per_row; ++i) {
        for (int j = 0; j < num_points_per_row; ++j) {
            double x = j * step;
            double y = i * step;
            points.emplace_back(x, y);
        }
    }

    // std::cout << "lenght of the points " << points.size();
    particles->resize(points.size());

    auto x_p_20 = particles->slicePosition();
    auto u_p_20 = particles->sliceVelocity();

    for (int i=0; i < u_p_20.size(); i++){
      x_p_20( i, 0 ) = points[i].first;
      x_p_20( i, 1 ) = points[i].second;
      x_p_20( i, 2 ) = 0.;
    }
}


double ComputeNeighbours()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana compute neighbours example\n" << std::endl;


  // ====================================================
  //             Use default Kokkos spaces
  // ====================================================
  using exec_space = Kokkos::OpenMP;
  using memory_space = typename exec_space::memory_space;


  // ====================================================
  //                 Particle generation
  // ====================================================
  auto particles = std::make_shared<
    CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1);
  // create points on a grid
  create_points(particles);

  double grid_min[3] = { -0.3, -0.3, -0.3 };
  double grid_max[3] = { 1.3, 1.3, 0.3 };

  double neighborhood_radius = 0.1005;
  double cell_ratio = 1.0;
  using ListAlgorithm = Cabana::FullNeighborTag;
  using ListType =
    Cabana::VerletList<memory_space, ListAlgorithm,
                       Cabana::VerletLayoutCSR,
                       Cabana::TeamOpTag>;

  auto positions = particles->slicePosition();
  auto forces = particles->sliceForce();
  ListType verlet_list( positions, 0, positions.size(), neighborhood_radius,
                        cell_ratio, grid_min, grid_max );

  auto first_neighbor_kernel = KOKKOS_LAMBDA( const int i, const int j )
    {

      auto dx = positions( i, 0 ) - positions( j, 0 );
      auto dy = positions( i, 0 ) - positions( j, 0 );
      auto dz = positions( i, 0 ) - positions( j, 0 );
      auto dist = sqrt(dx*dx + dy*dz + dz*dz);
      forces (i, 0) += 1;
    };

  Kokkos::RangePolicy<exec_space> policy( 0, positions.size() );

  Cabana::neighbor_parallel_for( policy, first_neighbor_kernel, verlet_list,
                                 Cabana::FirstNeighborsTag(),
                                 Cabana::SerialOpTag(), "ex_1st_serial" );
  Kokkos::fence();

  std::cout << "Cabana::neighbor_parallel_for results (first, serial)"
            << std::endl;
  for ( std::size_t i = 0; i < positions.size(); i++ )
    std::cout << forces( i, 0 ) << " ";
  std::cout << std::endl << std::endl;

  return 0;
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );
  ComputeNeighbours();

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
