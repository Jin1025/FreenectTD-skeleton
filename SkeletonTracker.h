//
//  SkeletonTracker.h
//  FreenectTD
//
//  BlazePose-based multi-person skeleton tracking via ONNX Runtime.
//  Outputs 33 keypoints per person (MediaPipe BlazePose format).
//

#pragma once

#include <array>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>

// Forward-declare ONNX Runtime types to avoid including heavy headers here
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct MemoryInfo;
}

struct Joint {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float visibility = 0.f;
};

struct TrackedPerson {
    int id = 0;
    std::array<Joint, 33> joints{};
    float confidence = 0.f;
    float bbox[4] = {0, 0, 0, 0}; // cx, cy, w, h (normalized 0-1)
    bool active = false;
};

static constexpr int kNumJoints = 33;
static constexpr int kMaxPlayers = 6;
static constexpr int kDetectorInputSize = 224;
static constexpr int kLandmarkInputSize = 256;
static constexpr int kNumAnchors = 2254;

static const char* const kJointNames[kNumJoints] = {
    "nose", "eye_inner_left", "eye_left", "eye_outer_left",
    "eye_inner_right", "eye_right", "eye_outer_right",
    "ear_left", "ear_right", "mouth_left", "mouth_right",
    "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
    "wrist_left", "wrist_right", "pinky_left", "pinky_right",
    "index_left", "index_right", "thumb_left", "thumb_right",
    "hip_left", "hip_right", "knee_left", "knee_right",
    "ankle_left", "ankle_right", "heel_left", "heel_right",
    "foot_index_left", "foot_index_right"
};

class SkeletonTracker {
public:
    SkeletonTracker();
    ~SkeletonTracker();

    SkeletonTracker(const SkeletonTracker&) = delete;
    SkeletonTracker& operator=(const SkeletonTracker&) = delete;

    bool initialize(const std::string& modelDir, int maxPlayers = 1);
    void shutdown();
    bool isInitialized() const { return initialized.load(); }

    void setMaxPlayers(int n);
    int  getMaxPlayers() const { return maxPlayers; }

    // Feed an RGB frame. Non-blocking: copies data and returns immediately.
    // Inference runs asynchronously. Results available via getPerson().
    void submitFrame(const uint8_t* rgbaData, int width, int height);

    int getNumTrackedPersons() const;
    TrackedPerson getPerson(int index) const;

    void setDetectorInterval(int n) { detectorInterval = std::max(1, n); }

private:
    // ONNX Runtime internals (pimpl to hide Ort:: types from header consumers)
    struct OrtState;
    OrtState* ort = nullptr;

    std::atomic<bool> initialized{false};
    int maxPlayers = 1;
    int detectorInterval = 5;

    // Double-buffered results
    std::vector<TrackedPerson> results;       // read by main thread
    std::vector<TrackedPerson> pendingResults; // written by infer thread
    mutable std::mutex resultMutex;

    // Frame submission
    std::vector<uint8_t> submittedFrame;
    int submittedWidth = 0;
    int submittedHeight = 0;
    bool hasNewFrame = false;
    std::mutex frameMutex;

    // Inference thread
    std::thread inferThread;
    std::atomic<bool> stopInfer{true};
    void inferLoop();

    // Pipeline internals
    int frameCount = 0;
    std::vector<std::array<float, 4>> lastDetections; // bboxes from last detection run
    int nextPersonId = 1;

    // SSD anchors for pose detection model
    std::vector<std::array<float, 2>> anchors; // [cx, cy] for each of 2254 anchors
    void generateAnchors();

    // Pipeline stages
    struct Detection {
        float cx, cy, w, h, score;
    };
    std::vector<Detection> runDetector(const std::vector<float>& rgbNormalized, int srcW, int srcH);
    std::array<Joint, kNumJoints> runLandmark(const std::vector<float>& rgbNormalized,
                                               int srcW, int srcH,
                                               const Detection& det,
                                               float& outConfidence);

    // Image preprocessing helpers
    static std::vector<float> rgbaToNormalizedRgb(const uint8_t* rgba, int w, int h);
    static std::vector<float> cropAndResize(const std::vector<float>& rgb, int srcW, int srcH,
                                            float cx, float cy, float size, int dstSize);

    // Non-Maximum Suppression
    static std::vector<Detection> nms(std::vector<Detection>& dets, float iouThresh);
    static float computeIoU(const Detection& a, const Detection& b);

    // Person ID tracking across frames
    void assignPersonIds(std::vector<TrackedPerson>& persons);

    static const TrackedPerson kEmptyPerson;
};
