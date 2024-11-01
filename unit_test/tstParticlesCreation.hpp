/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaNewPkg.hpp>

namespace Test
{
//---------------------------------------------------------------------------//
void testParticlesCreation( int steps )
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = typename exec_space::memory_space;

    auto particles = std::make_shared<
      CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1, "test_integrator");
    particles->resize(20);
    auto x_p = particles->slicePosition();
    EXPECT_DOUBLE_EQ(20, x_p.size());
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_integrate_reversibility )
{
    testParticlesCreation( 100 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
