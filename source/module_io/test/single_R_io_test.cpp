#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "module_io/single_R_io.h"
#include "module_base/global_variable.h"
#include "module_basis/module_ao/parallel_orbitals.h"
/************************************************
 *  unit test of output_single_R
 ***********************************************/
/**
 * - Tested Functions:
 *   - ModuleIO::output_single_R
 *     - output single R data
 */
Parallel_Orbitals::Parallel_Orbitals()
{
	trace_loc_row = nullptr;
}

Parallel_Orbitals::~Parallel_Orbitals()
{
	delete[] trace_loc_row;
}

TEST(ModuleIOTest, OutputSingleR)
{
    // Create temporary output file
    std::stringstream ofs_filename;
    GlobalV::DRANK=0;
    ofs_filename << "test_output_single_R_" << GlobalV::DRANK << ".dat";
    std::ofstream ofs(ofs_filename.str());

    // Define input parameters
    const double sparse_threshold = 1e-8;
    const bool binary = false;
    Parallel_Orbitals pv;
    GlobalV::NLOCAL=5;
    pv.trace_loc_row = new int[GlobalV::NLOCAL];
    pv.trace_loc_row[0] = 0;
    pv.trace_loc_row[1] = 1;
    pv.trace_loc_row[2] = -1;
    pv.trace_loc_row[3] = 2;
    pv.trace_loc_row[4] = -1; //Some rows have trace_loc_row < 0
    std::map<size_t, std::map<size_t, double>> XR = {
        {0, {{1, 0.5}, {3, 0.3}}},
        {1, {{0, 0.2}, {2, 0.4}}},
        {3, {{1, 0.1}, {4, 0.7}}}
    };

    // Call function under test
    ModuleIO::output_single_R(ofs, XR, sparse_threshold, binary, pv);

    // Close output file and open it for reading
    ofs.close();
    std::ifstream ifs;
    ifs.open("test_output_single_R_0.dat");
    std::string str((std::istreambuf_iterator<char>(ifs)),std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("5.00000000e-01 3.00000000e-01 2.00000000e-01 4.00000000e-01 1.00000000e-01 7.00000000e-01"));
    EXPECT_THAT(str, testing::HasSubstr("1 3 0 2 1 4"));
    EXPECT_THAT(str, testing::HasSubstr("0 2 4 4 6 6"));
    std::remove("test_output_single_R_0.dat");
}

int main(int argc, char **argv)
{

#ifdef __MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD,&GlobalV::MY_RANK);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    return result;
}