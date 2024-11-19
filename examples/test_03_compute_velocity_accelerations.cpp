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
  using ListType = Cabana::VerletList<memory_space, ListAlgorithm,
                                      Cabana::VerletLayoutCSR,
                                      Cabana::TeamOpTag>;

  auto positions = particles->slicePosition();
  auto verlet_list = std::make_shared<ListType>( positions, 0, positions.size(), neighborhood_radius,
                        cell_ratio, grid_min, grid_max );

  // create the accelerations object
  auto vel_acc = std::make_shared<CabanaNewPkg::VelocityAccelerations<exec_space>>( 0.1, 0.2 );


  auto x = particles->slicePosition();
  auto u = particles->sliceVelocity();
  auto force = particles->sliceForce();
  auto m = particles->sliceMass();
  auto rad = particles->sliceRadius();
  auto k = particles->sliceStiffness();

  Cabana::deep_copy( force, 10. );
  Cabana::deep_copy( m, 1. );
  Cabana::deep_copy( rad, 0.1 );
  Cabana::deep_copy( k, 100. );
  for (int i=0; i < positions.size(); i++){
    std::cout << force ( i, 0 ) << ", " << force ( i, 1 ) << ", " << force ( i, 2 ) << "\n";
  }
  // now compute the forces
  vel_acc->makeForceTorqueZeroOnParticle(*particles);
  CabanaNewPkg::computeForceParticleParticle(*vel_acc, *particles, *verlet_list);

  for (int i=0; i < positions.size(); i++){
    std::cout << force ( i, 0 ) << ", " << force ( i, 1 ) << ", " << force ( i, 2 ) << "\n";
  }

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
