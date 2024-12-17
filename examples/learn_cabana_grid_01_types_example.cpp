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
// Types example.
//---------------------------------------------------------------------------//
void typesExample()
{
    Cabana::Grid::UniformMesh<double> uniform;
    std::cout << "Uniform mesh with dimension " << uniform.num_space_dim
              << std::endl;

    Cabana::Grid::NonUniformMesh<double> nonuniform_double;
    std::cout << "Non-Uniform mesh with dimension " << nonuniform_double.num_space_dim
              << std::endl;

    Cabana::Grid::NonUniformMesh<float, 2> nonuniform_float;
    std::cout << "Non-Uniform mesh with dimension " << nonuniform_float.num_space_dim
              << std::endl;

    Cabana::Grid::SparseMesh<double, 3> sparse;
    std::cout << "Sparse mesh with dimension " << sparse.num_space_dim
              << std::endl;


    std::cout << "Is Cell a Cell? "
              << Cabana::Grid::isCell<Cabana::Grid::Cell>() << std::endl;
    std::cout << "Is Node a Node? "
              << Cabana::Grid::isNode<Cabana::Grid::Node>() << std::endl;
    std::cout << "Is Cell a Node? "
              << Cabana::Grid::isNode<Cabana::Grid::Cell>() << "\n"
              << std::endl;

    std::cout
        << "Is I Edge an Edge? "
        << Cabana::Grid::isEdge<Cabana::Grid::Edge<Cabana::Grid::Dim::I>>()
        << std::endl;
    std::cout
        << "Is J Face a Face? "
        << Cabana::Grid::isFace<Cabana::Grid::Face<Cabana::Grid::Dim::J>>()
        << std::endl;
    std::cout
        << "Is K Face an Edge? "
        << Cabana::Grid::isEdge<Cabana::Grid::Face<Cabana::Grid::Dim::K>>()
        << "\n"
        << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    typesExample();

    return 0;
}

//---------------------------------------------------------------------------//
