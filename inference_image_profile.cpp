//
// Profiling executable for SuperPoint + SuperGlue TensorRT
// Tests extraction and matching at multiple resolutions (1.0x, 0.5x, 0.25x)
// and prints a detailed timing breakdown table.
//

#include <memory>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include "utils.h"
#include "super_glue.h"
#include "super_point.h"

// ─── Timing helpers ──────────────────────────────────────────────────────────

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Ms        = std::chrono::duration<double, std::milli>;

static double elapsed_ms(TimePoint a, TimePoint b) {
    return std::chrono::duration_cast<Ms>(b - a).count();
}

struct Stats {
    double mean{}, min_v{}, max_v{}, stddev{};
    int    count{};
};

static Stats compute_stats(const std::vector<double>& v) {
    if (v.empty()) return {};
    Stats s;
    s.count = (int)v.size();
    s.mean  = std::accumulate(v.begin(), v.end(), 0.0) / s.count;
    s.min_v = *std::min_element(v.begin(), v.end());
    s.max_v = *std::max_element(v.begin(), v.end());
    double sq = 0;
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / s.count);
    return s;
}

// ─── Pretty printing ─────────────────────────────────────────────────────────

static void print_separator(int w = 90) {
    std::cout << std::string(w, '-') << "\n";
}

static void print_header() {
    print_separator();
    std::cout << std::left
              << std::setw(18) << "Resolution"
              << std::setw(12) << "W x H"
              << std::setw(10) << "Kpts0"
              << std::setw(10) << "Kpts1"
              << std::setw(10) << "Matches"
              << std::setw(14) << "SP0 ms(mean)"
              << std::setw(14) << "SP1 ms(mean)"
              << std::setw(14) << "SG  ms(mean)"
              << std::setw(12) << "Total ms"
              << "\n";
    print_separator();
}

static void print_stats_block(const std::string& label, const Stats& s) {
    std::cout << "    " << std::left << std::setw(24) << label
              << "mean=" << std::fixed << std::setprecision(2) << std::setw(8) << s.mean
              << "  min=" << std::setw(8) << s.min_v
              << "  max=" << std::setw(8) << s.max_v
              << "  std=" << std::setw(8) << s.stddev
              << "  (n=" << s.count << ")\n";
}

// ─── Per-resolution profiling run ────────────────────────────────────────────

struct RunResult {
    std::string label;
    int width, height;
    int kpts0{}, kpts1{}, matches{};
    Stats sp0, sp1, sg, total;
};

static RunResult profile_resolution(
        const std::shared_ptr<SuperPoint>& superpoint,
        const std::shared_ptr<SuperGlue>&  superglue,
        const cv::Mat& image0_orig,
        const cv::Mat& image1_orig,
        int target_w, int target_h,
        const std::string& label,
        int warmup  = 3,
        int repeats = 20)
{
    cv::Mat img0, img1;
    cv::resize(image0_orig, img0, cv::Size(target_w, target_h));
    cv::resize(image1_orig, img1, cv::Size(target_w, target_h));

    Eigen::Matrix<double, 259, Eigen::Dynamic> fp0, fp1;
    std::vector<cv::DMatch> matches;

    std::vector<double> t_sp0, t_sp1, t_sg, t_total;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        superpoint->infer(img0, fp0);
        superpoint->infer(img1, fp1);
        superglue->matching_points(fp0, fp1, matches, false, target_w, target_h);
    }

    // Timed runs
    for (int i = 0; i < repeats; ++i) {
        auto t0 = Clock::now();

        auto ta = Clock::now();
        superpoint->infer(img0, fp0);
        auto tb = Clock::now();

        superpoint->infer(img1, fp1);
        auto tc = Clock::now();

        superglue->matching_points(fp0, fp1, matches, false, target_w, target_h);
        auto td = Clock::now();

        t_sp0.push_back(elapsed_ms(ta, tb));
        t_sp1.push_back(elapsed_ms(tb, tc));
        t_sg.push_back(elapsed_ms(tc, td));
        t_total.push_back(elapsed_ms(t0, td));
    }

    RunResult r;
    r.label   = label;
    r.width   = target_w;
    r.height  = target_h;
    r.kpts0   = (int)fp0.cols();
    r.kpts1   = (int)fp1.cols();
    r.matches = (int)matches.size();
    r.sp0     = compute_stats(t_sp0);
    r.sp1     = compute_stats(t_sp1);
    r.sg      = compute_stats(t_sg);
    r.total   = compute_stats(t_total);
    return r;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "./superpointglue_profile config_path model_dir "
                     "first_image_path second_image_path\n";
        return 1;
    }

    std::string config_path  = argv[1];
    std::string model_dir    = argv[2];
    std::string image0_path  = argv[3];
    std::string image1_path  = argv[4];

    // Load images at original resolution
    cv::Mat image0_orig = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image1_orig = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    if (image0_orig.empty() || image1_orig.empty()) {
        std::cerr << "Failed to load input images.\n";
        return 1;
    }

    int orig_w = image0_orig.cols;
    int orig_h = image0_orig.rows;
    std::cout << "Original image size: " << orig_w << " x " << orig_h << "\n";

    // Build engines at the config resolution (needed for SuperGlue normalization)
    Configs configs(config_path, model_dir);

    std::cout << "Building inference engines...\n";
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error building SuperPoint engine.\n";
        return 1;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error building SuperGlue engine.\n";
        return 1;
    }
    std::cout << "Engines ready.\n\n";

    // Define resolutions to test
    struct Scale { std::string label; double factor; };
    std::vector<Scale> scales = {
        {"1.00x (original)", 1.0},
        {"0.75x",            0.75},
        {"0.50x",            0.5},
        {"0.25x",            0.25},
    };

    // Also include the config resolution if different from original
    int cfg_w = configs.superglue_config.image_width;
    int cfg_h = configs.superglue_config.image_height;
    bool cfg_is_custom = (cfg_w != orig_w || cfg_h != orig_h);

    const int WARMUP  = 3;
    const int REPEATS = 20;

    std::vector<RunResult> results;

    // Config resolution first (always run)
    std::cout << "Profiling config resolution (" << cfg_w << "x" << cfg_h << ")...\n";
    results.push_back(profile_resolution(
        superpoint, superglue, image0_orig, image1_orig,
        cfg_w, cfg_h,
        "config (" + std::to_string(cfg_w) + "x" + std::to_string(cfg_h) + ")",
        WARMUP, REPEATS));

    // Scale-based resolutions
    for (auto& sc : scales) {
        int w = std::max(32, (int)(orig_w * sc.factor));
        int h = std::max(32, (int)(orig_h * sc.factor));
        // Skip if this duplicates the config resolution
        if (w == cfg_w && h == cfg_h) continue;
        std::cout << "Profiling " << sc.label << " (" << w << "x" << h << ")...\n";
        results.push_back(profile_resolution(
            superpoint, superglue, image0_orig, image1_orig,
            w, h, sc.label, WARMUP, REPEATS));
    }

    // ── Summary table ──────────────────────────────────────────────────────
    std::cout << "\n\n";
    std::cout << "=========================================================="
                 "==========================\n";
    std::cout << "  PROFILING SUMMARY  —  SuperPoint + SuperGlue TensorRT\n";
    std::cout << "  Images: " << image0_path << "  |  " << image1_path << "\n";
    std::cout << "  Warmup: " << WARMUP << " runs   Timed: " << REPEATS << " runs\n";
    std::cout << "=========================================================="
                 "==========================\n\n";

    print_header();
    for (auto& r : results) {
        std::cout << std::left
                  << std::setw(18) << r.label
                  << std::setw(12) << (std::to_string(r.width) + "x" + std::to_string(r.height))
                  << std::setw(10) << r.kpts0
                  << std::setw(10) << r.kpts1
                  << std::setw(10) << r.matches
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.sp0.mean
                  << std::setw(14) << r.sp1.mean
                  << std::setw(14) << r.sg.mean
                  << std::setw(12) << r.total.mean
                  << "\n";
    }
    print_separator();

    // ── Detailed per-resolution breakdown ─────────────────────────────────
    std::cout << "\n  DETAILED BREAKDOWN\n";
    for (auto& r : results) {
        std::cout << "\n  [ " << r.label << "  —  "
                  << r.width << "x" << r.height << " ]\n";
        std::cout << "  Keypoints: img0=" << r.kpts0
                  << "  img1=" << r.kpts1
                  << "  Matches=" << r.matches
                  << "  MatchRate=" << std::fixed << std::setprecision(1)
                  << (r.kpts0 > 0 ? 100.0 * r.matches / r.kpts0 : 0.0) << "%\n";
        print_stats_block("SuperPoint img0 (ms)", r.sp0);
        print_stats_block("SuperPoint img1 (ms)", r.sp1);
        print_stats_block("SuperGlue match (ms)", r.sg);
        print_stats_block("Total pipeline  (ms)", r.total);
        std::cout << "  Throughput: "
                  << std::fixed << std::setprecision(1)
                  << (1000.0 / r.total.mean) << " pairs/sec  |  "
                  << std::setprecision(1)
                  << (2000.0 / r.total.mean) << " images/sec\n";
    }

    // ── Save match images at each resolution ──────────────────────────────
    std::cout << "\n  Saving match images...\n";
    for (auto& r : results) {
        cv::Mat img0, img1;
        cv::resize(image0_orig, img0, cv::Size(r.width, r.height));
        cv::resize(image1_orig, img1, cv::Size(r.width, r.height));

        Eigen::Matrix<double, 259, Eigen::Dynamic> fp0, fp1;
        superpoint->infer(img0, fp0);
        superpoint->infer(img1, fp1);

        std::vector<cv::DMatch> matches;
        superglue->matching_points(fp0, fp1, matches, false, r.width, r.height);

        std::vector<cv::KeyPoint> kp0, kp1;
        for (int i = 0; i < fp0.cols(); ++i)
            kp0.emplace_back(fp0(1,i), fp0(2,i), 8, -1, fp0(0,i));
        for (int i = 0; i < fp1.cols(); ++i)
            kp1.emplace_back(fp1(1,i), fp1(2,i), 8, -1, fp1(0,i));

        cv::Mat out;
        cv::drawMatches(img0, kp0, img1, kp1, matches, out,
                        cv::Scalar(0,255,0), cv::Scalar(0,0,255));

        // Overlay label
        std::string text = r.label + "  matches=" + std::to_string((int)matches.size());
        cv::putText(out, text, cv::Point(8, 20),
                    cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0,0,0), 2, cv::LINE_AA);
        cv::putText(out, text, cv::Point(8, 20),
                    cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255,255,255), 1, cv::LINE_AA);

        // Sanitise label for filename
        std::string fname = r.label;
        for (char& c : fname) if (c == ' ' || c == '(' || c == ')') c = '_';
        fname = "match_" + fname + ".png";
        cv::imwrite(fname, out);
        std::cout << "  Saved: " << fname << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
