#ifndef CabanaNewPkgVelocityAccelerations_HPP
#define CabanaNewPkgVelocityAccelerations_HPP

#include <cmath>

#include <CabanaNewPkg_Particles.hpp>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaNewPkg
{
  template <class ExecutionSpace>
  class VelocityAccelerations
  {
    using exec_space = ExecutionSpace;

  public:
    double eta, beta;

    VelocityAccelerations(double eta, double beta):
      eta (eta),
      beta (beta) {

    }

    ~VelocityAccelerations() {}

    template <class ParticleType>
    void makeForceTorqueZeroOnParticle(ParticleType& particles)
    {
      auto force = particles.sliceForce();

      Cabana::deep_copy( force, 0. );
    }

    template <class ParticleType, class NeighListType>
    void computeForceFullParticleParticle(ParticleType& particles,
                                          const NeighListType& neigh_list)
    {
      auto x = particles.slicePosition();
      auto u = particles.sliceVelocity();
      auto force = particles.sliceForce();
      auto m = particles.sliceMass();
      auto rad = particles.sliceRadius();
      auto k = particles.sliceStiffness();

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

          // const double mass_i = aosoa_mass( i );
          const double mass_j = m ( j );

          // Find the overlap amount
          double overlap =  rad ( i ) + rad ( j ) - rij;

          double a_i = rad ( i ) - overlap / 2.;
          double a_j = rad ( j ) - overlap / 2.;

          // normal vector passing from j to i
          double nij_x = pos_ij[0] / rij;
          double nij_y = pos_ij[1] / rij;
          double nij_z = pos_ij[2] / rij;

          double vel_i[3] = {0., 0., 0.};
          vel_i[0] = u ( i, 0 );
          vel_i[1] = u ( i, 1 );
          vel_i[2] = u ( i, 2 );

          double vel_j[3] = {0., 0., 0.};
          vel_j[0] = u ( j, 0 );
          vel_j[1] = u ( j, 1 );
          vel_j[2] = u ( j, 2 );

          // Now the relative velocity of particle i w.r.t j at the contact
          // point is
          double vel_ij[3] = {vel_i[0] - vel_j[0],
            vel_i[1] - vel_j[1],
            vel_i[2] - vel_j[2]};

          // normal velocity magnitude
          double vij_dot_nij = vel_ij[0] * nij_x + vel_ij[1] * nij_y + vel_ij[2] * nij_z;
          double vn_x = vij_dot_nij * nij_x;
          double vn_y = vij_dot_nij * nij_y;
          double vn_z = vij_dot_nij * nij_z;

          // tangential velocity
          double vt_x = vel_ij[0] - vn_x;
          double vt_y = vel_ij[1] - vn_y;
          double vt_z = vel_ij[2] - vn_z;

          /*
            ====================================
            End: common to all equations in SPH.
            ====================================
          */
          // find the force if the particles are overlapping
          if (overlap > 0.) {
            double k_ij = 0.5 * (k( i ) + k( j ));

            // normal force
            double fn =  k_ij * overlap;
            double fn_x = fn * nij_x;
            double fn_y = fn * nij_y;
            double fn_z = fn * nij_z;

            // Add force to the particle i due to contact with particle j
            force( i, 0 ) += fn_x;
            force( i, 1 ) += fn_y;
            force( i, 2 ) += fn_z;
          }
        };

      Kokkos::RangePolicy<exec_space> policy(0, u.size());

      Cabana::neighbor_parallel_for( policy,
                                     force_full,
                                     neigh_list,
                                     Cabana::FirstNeighborsTag(),
                                     Cabana::SerialOpTag(),
                                     "CabanaDEM::ForceFull" );
      Kokkos::fence();
    }

  };
  /******************************************************************************
  Force functions
  ******************************************************************************/

  template <class AccelerationType, class ParticleType, class NeighListType>
  void computeForceParticleParticle( AccelerationType& accelerations, ParticleType& particles,
                                     const NeighListType& neigh_list)
  {
    accelerations.makeForceTorqueZeroOnParticle( particles );
    accelerations.computeForceFullParticleParticle( particles, neigh_list );
  }

  // template <class ParticleType, class NeighListType, class ExecutionSpace>
  template <class ParticleType, class NeighListType>
  void testNestedLoopEquation(ParticleType& particles, const NeighListType& neigh_list,
                              double neighbour_radius)
  {
      auto x = particles.slicePosition();
      auto rad = particles.sliceRadius();

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

          if (rij < neighbour_radius) {
            rad( i ) += 1.;
          }
        };

      Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, x.size());

      Cabana::neighbor_parallel_for( policy,
                                     force_full,
                                     neigh_list,
                                     Cabana::FirstNeighborsTag(),
                                     Cabana::SerialOpTag(),
                                     "CabanaDEM::ForceFull" );
      Kokkos::fence();
  }

  // template <class ParticleType, class NeighListType, class ExecutionSpace>
  template <class ParticleType, class NeighListType>
  void testReduceEquation(ParticleType& particles, const NeighListType& neigh_list,
                          double neighbour_radius, double & reduce_)
  {
      auto x = particles.slicePosition();
      auto rad = particles.sliceRadius();

      auto reduce_op =
        KOKKOS_LAMBDA( const std::size_t i, double& reduce_ )
        {
          reduce_ += rad( i );
        };

      Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, x.size());

      Kokkos::parallel_reduce("reduce_serial", policy, reduce_op, reduce_ );
      Kokkos::fence();
  }

  // template <class ParticleType, class NeighListType, class ExecutionSpace>
  template <class ParticleType, class NeighListType>
  void testSerialLoopEquation(ParticleType& particles, const NeighListType& neigh_list,
                              double neighbour_radius)
  {
      auto x = particles.slicePosition();
      auto rad = particles.sliceRadius();

      auto move_x = KOKKOS_LAMBDA( const int i )
        {
          x( i, 0 ) += 0.5;
        };

      Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, x.size());

      Kokkos::parallel_for( "CabanaNewPkg::MoveX", policy,
                            move_x);
  }

}

#endif
