/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Grid.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// Partitioner example.
//---------------------------------------------------------------------------//
void partitionerExample()
{
  int comm_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  if ( comm_rank == 0 )
    {
      std::cout << "Cabana::Grid Partitioner Example" << std::endl;
      std::cout << "     (intended to be run with MPI)\n" << std::endl;
    }

  // ===================
  // Automatic partition
  // ===================
  Cabana::Grid::DimBlockPartitioner<3> dim_block_partitioner;

  std::array<int, 3> ranks_per_dim_block =
    dim_block_partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0, 0 } );

  // Print the created decomposition.
  if ( comm_rank == 0 )
    {
      std::cout << "Ranks per dimension (automatic): ";
      for ( int d = 0; d < 3; ++d )
        std::cout << ranks_per_dim_block[d] << " ";
      std::cout << std::endl;
    }
  // ========================
  // Automatic partition ends
  // ========================
  // ===================
  // Manual partition
  // ===================
  int comm_size;
  MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
  std::array<int, 2> input_ranks_per_dim = { comm_size, 1 };

  Cabana::Grid::ManualBlockPartitioner<2> manual_partitioner(input_ranks_per_dim);

  std::array<int, 2> ranks_per_dim_manual =
    manual_partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0 } );

  // Print the created decomposition.
  if ( comm_rank == 0 )
    {
      std::cout << "Ranks per dimension (automatic): ";
      for ( int d = 0; d < 2; ++d )
        std::cout << ranks_per_dim_manual[d] << " ";
      std::cout << std::endl;
    }
  // ========================
  // Manual partition ends
  // ========================
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv);
  {
    Kokkos::ScopeGuard scope_guard ( argc, argv);
    partitionerExample();
  }
  MPI_Finalize();
  return 0;
}
