#pragma once
#include <queue>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <fstream>
#include <iostream>
#include <arm_neon.h>

// NEON 内积：4路展开
inline float dot_neon(const float* __restrict__ a,
                      const float* __restrict__ b,
                      size_t dim) {
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);
    size_t d = 0;
    for (; d + 15 < dim; d += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a+d),    vld1q_f32(b+d));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a+d+4),  vld1q_f32(b+d+4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a+d+8),  vld1q_f32(b+d+8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a+d+12), vld1q_f32(b+d+12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    for (; d + 3 < dim; d += 4)
        acc0 = vmlaq_f32(acc0, vld1q_f32(a+d), vld1q_f32(b+d));
    float32x2_t s2 = vadd_f32(vget_high_f32(acc0), vget_low_f32(acc0));
    float res = vget_lane_f32(vpadd_f32(s2, s2), 0);
    for (; d < dim; ++d) res += a[d] * b[d];
    return res;
}

// NEON L2 距离平方：仅用于 K-Means 聚类
inline float l2sq_neon(const float* __restrict__ a,
                       const float* __restrict__ b,
                       size_t dim) {
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    size_t d = 0;
    for (; d + 7 < dim; d += 8) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a+d),   vld1q_f32(b+d));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a+d+4), vld1q_f32(b+d+4));
        acc0 = vmlaq_f32(acc0, d0, d0);
        acc1 = vmlaq_f32(acc1, d1, d1);
    }
    acc0 = vaddq_f32(acc0, acc1);
    for (; d + 3 < dim; d += 4) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a+d), vld1q_f32(b+d));
        acc0 = vmlaq_f32(acc0, d0, d0);
    }
    float32x2_t s2 = vadd_f32(vget_high_f32(acc0), vget_low_f32(acc0));
    float res = vget_lane_f32(vpadd_f32(s2, s2), 0);
    for (; d < dim; ++d) { float diff=a[d]-b[d]; res += diff*diff; }
    return res;
}

// K-Means：训练和编码都用 L2（标准 PQ）
static void kmeans(const float* data, size_t n, size_t dsub,
                   size_t Ks, int max_iter, float* centers) {
    std::mt19937 rng(42);
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    for (size_t i = 0; i < Ks && i < n; ++i) {
        size_t j = i + rng() % (n - i);
        std::swap(idx[i], idx[j]);
        memcpy(centers + i*dsub, data + idx[i]*dsub, dsub*sizeof(float));
    }
    std::vector<uint32_t> assign(n);
    std::vector<float>    new_centers(Ks * dsub);
    std::vector<uint32_t> cnt(Ks);
    for (int it = 0; it < max_iter; ++it) {
        for (size_t i = 0; i < n; ++i) {
            float best = std::numeric_limits<float>::max();
            uint32_t best_k = 0;
            for (size_t k = 0; k < Ks; ++k) {
                float d = l2sq_neon(data+i*dsub, centers+k*dsub, dsub);
                if (d < best) { best = d; best_k = (uint32_t)k; }
            }
            assign[i] = best_k;
        }
        memset(new_centers.data(), 0, Ks*dsub*sizeof(float));
        memset(cnt.data(), 0, Ks*sizeof(uint32_t));
        for (size_t i = 0; i < n; ++i) {
            uint32_t k = assign[i];
            cnt[k]++;
            for (size_t d = 0; d < dsub; ++d)
                new_centers[k*dsub+d] += data[i*dsub+d];
        }
        for (size_t k = 0; k < Ks; ++k) {
            if (cnt[k] > 0) {
                float inv = 1.f / cnt[k];
                for (size_t d = 0; d < dsub; ++d)
                    centers[k*dsub+d] = new_centers[k*dsub+d] * inv;
            }
        }
    }
}

struct PQIndex {
    size_t vecdim, M, Ks, dsub;
    std::vector<float>   codebook;
    std::vector<uint8_t> codes;
    size_t base_number;

    PQIndex(size_t vecdim_, size_t M_=8, size_t Ks_=256)
        : vecdim(vecdim_), M(M_), Ks(Ks_),
          dsub(vecdim_/M_), base_number(0) {
        codebook.resize(M*Ks*dsub, 0.f);
    }

    bool save_codebook(const std::string& path) const {
        std::ofstream fout(path, std::ios::binary);
        if (!fout) { std::cerr<<"[PQ] 无法写入: "<<path<<"\n"; return false; }
        uint64_t hdr[3]={(uint64_t)M,(uint64_t)Ks,(uint64_t)dsub};
        fout.write((char*)hdr, sizeof(hdr));
        fout.write((char*)codebook.data(), codebook.size()*sizeof(float));
        std::cerr<<"[PQ] 码本已保存: "<<path<<"\n";
        return true;
    }

    bool load_codebook(const std::string& path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) return false;
        uint64_t hdr[3];
        fin.read((char*)hdr, sizeof(hdr));
        if (hdr[0]!=M||hdr[1]!=Ks||hdr[2]!=dsub) {
            std::cerr<<"[PQ] 参数不匹配，重新训练\n"; return false;
        }
        fin.read((char*)codebook.data(), codebook.size()*sizeof(float));
        std::cerr<<"[PQ] 码本加载成功: "<<path<<"\n";
        return true;
    }

    void load_or_train(const float* base, size_t n,
                       const std::string& path, int iters=25) {
        if (load_codebook(path)) return;
        std::cerr<<"[PQ] 开始训练 M="<<M<<" Ks="<<Ks<<" iter="<<iters<<"\n";
        std::vector<float> sub(n*dsub);
        for (size_t m = 0; m < M; ++m) {
            for (size_t i = 0; i < n; ++i)
                memcpy(sub.data()+i*dsub,
                       base+i*vecdim+m*dsub,
                       dsub*sizeof(float));
            kmeans(sub.data(), n, dsub, Ks, iters,
                   codebook.data()+m*Ks*dsub);
            std::cerr<<"[PQ] 子空间 "<<m+1<<"/"<<M<<" 完成\n";
        }
        save_codebook(path);
    }

    // 用 L2 找最近码字
    void encode(const float* base, size_t n) {
        base_number = n;
        codes.resize(n*M);
        for (size_t i = 0; i < n; ++i) {
            for (size_t m = 0; m < M; ++m) {
                const float* vsub = base + i*vecdim + m*dsub;
                const float* cb_m = codebook.data() + m*Ks*dsub;
                float best = std::numeric_limits<float>::max();
                uint8_t best_k = 0;
                for (size_t k = 0; k < Ks; ++k) {
                    float d = l2sq_neon(vsub, cb_m+k*dsub, dsub);
                    if (d < best) { best = d; best_k = (uint8_t)k; }
                }
                codes[i*M+m] = best_k;
            }
        }
        std::cerr<<"[PQ] 编码完成\n";
    }

    // 查询：ADC，LUT 用内积，dis = 1 - ip
    std::priority_queue<std::pair<float,uint32_t>>
    search(const float* query, size_t k) const {
        // 构建 LUT：query 子向量与每个码字的内积
        std::vector<float> lut(M*Ks);
        for (size_t m = 0; m < M; ++m) {
            const float* q_sub = query + m*dsub;
            const float* cb_m  = codebook.data() + m*Ks*dsub;
            float* lut_m = lut.data() + m*Ks;
            for (size_t kk = 0; kk < Ks; ++kk)
                lut_m[kk] = dot_neon(q_sub, cb_m+kk*dsub, dsub);
        }

        std::priority_queue<std::pair<float,uint32_t>> q;
        const uint8_t* c = codes.data();

        size_t i = 0;
        for (; i + 3 < base_number; i += 4) {
            float ip0=0, ip1=0, ip2=0, ip3=0;
            for (size_t m = 0; m < M; ++m) {
                const float* lut_m = lut.data() + m*Ks;
                ip0 += lut_m[c[(i+0)*M+m]];
                ip1 += lut_m[c[(i+1)*M+m]];
                ip2 += lut_m[c[(i+2)*M+m]];
                ip3 += lut_m[c[(i+3)*M+m]];
            }
            float32x4_t dis_vec = vsubq_f32(vdupq_n_f32(1.f),
                                  (float32x4_t){ip0,ip1,ip2,ip3});
            float dis[4]; vst1q_f32(dis, dis_vec);
            for (int t = 0; t < 4; ++t) {
                uint32_t id = (uint32_t)(i+t);
                if (q.size()<k) q.push({dis[t],id});
                else if (dis[t]<q.top().first){q.push({dis[t],id});q.pop();}
            }
        }
        for (; i < base_number; ++i) {
            float ip = 0;
            for (size_t m = 0; m < M; ++m)
                ip += lut.data()[m*Ks + c[i*M+m]];
            float dis = 1.f - ip;
            if (q.size()<k) q.push({dis,(uint32_t)i});
            else if (dis<q.top().first){q.push({dis,(uint32_t)i});q.pop();}
        }
        return q;
    }
};

std::priority_queue<std::pair<float,uint32_t>>
flat_search(const PQIndex& pq, const float* query,
            size_t base_number, size_t vecdim, size_t k) {
    return pq.search(query, k);
}
