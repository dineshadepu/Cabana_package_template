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
void testCreationOfVelocityAccelerationsObject()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = typename exec_space::memory_space;

    // Create the force object, here VelocityAccelerations
    auto vel_acc = CabanaNewPkg::VelocityAccelerations<exec_space>( 0.1, 0.2 );
    EXPECT_DOUBLE_EQ(0.1, vel_acc.eta);
    EXPECT_DOUBLE_EQ(0.2, vel_acc.beta);
}


void testComputationOfVelocityAccelerations()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = typename exec_space::memory_space;

    // create particles

    // Create the force object, here VelocityAccelerations
    auto vel_acc = CabanaNewPkg::VelocityAccelerations<exec_space>( 0.1, 0.2 );
    EXPECT_DOUBLE_EQ(0.1, vel_acc.eta);
    EXPECT_DOUBLE_EQ(0.2, vel_acc.beta);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_velocity_accelerations )
{
    testCreationOfVelocityAccelerationsObject();
}
//---------------------------------------------------------------------------//

} // end namespace Test
