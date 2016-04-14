#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <iostream>
#include "avx_mathfun.h"

constexpr size_t VECTOR_LEN = 1UL<<16;
constexpr size_t ALIGN = 32;
constexpr size_t ITER_COUNT = 1000;
constexpr float EPSILON = 1e-5;

class AVXMathfunTest : public ::testing::Test {
private:
    template <typename T>
    inline void allocMemory(T** data) {
        posix_memalign(reinterpret_cast<void**>(data), ALIGN, VECTOR_LEN*sizeof(float));
    }

protected:
    virtual void TearDown() {
//        std::cout<<"Tearing Down..."<<std::endl;
        free(inputData_);
        free(naiveExpResult_);
        free(simdExpResult_);

    }

    virtual void SetUp() {
//        std::cout<<"Setting Up..."<<std::endl;
        allocMemory(&inputData_);
        allocMemory(&naiveExpResult_);
        allocMemory(&simdExpResult_);


        std::random_device dev;
        std::mt19937_64 eng;
        eng.seed(dev());
        std::uniform_real_distribution<float> distribution(0, 1);

        for (size_t i = 0; i < VECTOR_LEN; ++i) {
            float tmp = distribution(eng);
            inputData_[i] = tmp;
            naiveExpResult_[i] = std::exp(tmp);
        }
    }

    float* inputData_;
    float* naiveExpResult_;
    float* simdExpResult_;

};

TEST_F(AVXMathfunTest, exp) {
    for (size_t i = 0; i < ITER_COUNT; ++i) {
        __m256 tmp;
        __m256 ipt;
        for (size_t j = 0; j < VECTOR_LEN; j+= 8) {
            ipt = _mm256_load_ps(inputData_ + j);
            tmp = exp256_ps(ipt);
            _mm256_store_ps(simdExpResult_ + j, tmp);
        }
    }

    for (size_t i = 0; i < VECTOR_LEN; ++i) {
        ASSERT_NEAR(naiveExpResult_[i], simdExpResult_[i], EPSILON);
    }
}