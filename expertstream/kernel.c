/* bf16 kernels. Weights stay in bf16 in memory, so a cached expert costs half of float32. */
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
static inline __m256 load8(const uint16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
#endif

static inline float one(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void bf16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = one(src[i]);
}

/* y = W x, W is rows x cols in bf16 */
void bf16_matvec(const uint16_t *w, int64_t rows, int64_t cols, const float *x, float *y) {
    for (int64_t r = 0; r < rows; r++, w += cols) {
        int64_t c = 0;
        float sum = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 a = _mm256_setzero_ps(), b = a, d = a, e = a;
        for (; c + 32 <= cols; c += 32) {
            a = _mm256_fmadd_ps(load8(w + c), _mm256_loadu_ps(x + c), a);
            b = _mm256_fmadd_ps(load8(w + c + 8), _mm256_loadu_ps(x + c + 8), b);
            d = _mm256_fmadd_ps(load8(w + c + 16), _mm256_loadu_ps(x + c + 16), d);
            e = _mm256_fmadd_ps(load8(w + c + 24), _mm256_loadu_ps(x + c + 24), e);
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(d, e)));
        for (int i = 0; i < 8; i++) sum += lanes[i];
#endif
        for (; c < cols; c++) sum += one(w[c]) * x[c];
        y[r] = sum;
    }
}

/* One SiLU-gated expert on one token. tmp holds 3*inner floats, y holds hidden floats. */
void expert_apply(const uint16_t *win, const uint16_t *wout, int64_t hidden, int64_t inner,
                  const float *x, float *tmp, float *y) {
    bf16_matvec(win, 2 * inner, hidden, x, tmp);
    float *gated = tmp + 2 * inner;
    for (int64_t i = 0; i < inner; i++)
        gated[i] = tmp[i] / (1.0f + expf(-tmp[i])) * tmp[inner + i];
    bf16_matvec(wout, hidden, inner, gated, y);
}
