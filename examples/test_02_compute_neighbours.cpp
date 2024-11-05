#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaNewPkg.hpp>

#define DIM 3


void create_points(auto & particles) {
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
  ListType verlet_list( positions, 0, positions.size(), neighborhood_radius,
                        cell_ratio, grid_min, grid_max );

  for ( std::size_t i = 0; i < positions.size(); ++i )
    {
      int num_n = Cabana::NeighborList<ListType>::numNeighbor( verlet_list, i );
      std::cout << "Particle " << i << " # neighbor = " << num_n << std::endl;
      for ( int j = 0; j < num_n; ++j )
        std::cout << "    neighbor " << j << " = "
                  << Cabana::NeighborList<ListType>::getNeighbor(verlet_list, i, j )
                  << std::endl;
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
