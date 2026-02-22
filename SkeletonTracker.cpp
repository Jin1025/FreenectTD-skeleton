//
//  SkeletonTracker.cpp
//  FreenectTD
//
//  BlazePose-based multi-person skeleton tracking via ONNX Runtime.
//

#include "SkeletonTracker.h"
#include "logger.h"

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/coreml_provider_factory.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstring>
#include <chrono>

// ───────────────────────────────────────────────────────────────
// OrtState: holds all ONNX Runtime objects (hidden from header)
// ───────────────────────────────────────────────────────────────

struct SkeletonTracker::OrtState {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "FreenectTD_Skeleton"};
    std::unique_ptr<Ort::Session> detectorSession;
    std::unique_ptr<Ort::Session> landmarkSession;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

const TrackedPerson SkeletonTracker::kEmptyPerson{};

// ───────────────────────────────────────────────────────────────
// Construction / Destruction
// ───────────────────────────────────────────────────────────────

SkeletonTracker::SkeletonTracker() = default;

SkeletonTracker::~SkeletonTracker() {
    shutdown();
}

// ───────────────────────────────────────────────────────────────
// Initialization
// ───────────────────────────────────────────────────────────────

bool SkeletonTracker::initialize(const std::string& modelDir, int maxP) {
    if (initialized.load()) return true;

    maxPlayers = std::clamp(maxP, 1, kMaxPlayers);
    results.resize(maxPlayers);
    pendingResults.resize(maxPlayers);

    try {
        ort = new OrtState();

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Try CoreML EP for Apple Neural Engine acceleration
        try {
            uint32_t coremlFlags = 0;
            // COREML_FLAG_USE_CPU_AND_GPU = 0x001 — let CoreML decide CPU vs GPU vs ANE
            coremlFlags |= 0x001;
            (void)OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coremlFlags);
            LOG("[SkeletonTracker] CoreML EP enabled");
        } catch (...) {
            LOG("[SkeletonTracker] CoreML EP not available, falling back to CPU");
        }

        std::string detectorPath = modelDir + "/pose_detection.onnx";
        std::string landmarkPath = modelDir + "/pose_landmark_full.onnx";

        ort->detectorSession = std::make_unique<Ort::Session>(ort->env, detectorPath.c_str(), opts);
        LOG("[SkeletonTracker] Detector model loaded: " + detectorPath);

        ort->landmarkSession = std::make_unique<Ort::Session>(ort->env, landmarkPath.c_str(), opts);
        LOG("[SkeletonTracker] Landmark model loaded: " + landmarkPath);

    } catch (const Ort::Exception& e) {
        LOG("[SkeletonTracker] ONNX init failed: " + std::string(e.what()));
        delete ort;
        ort = nullptr;
        return false;
    }

    generateAnchors();
    initialized = true;

    stopInfer = false;
    inferThread = std::thread(&SkeletonTracker::inferLoop, this);

    LOG("[SkeletonTracker] Initialized with maxPlayers=" + std::to_string(maxPlayers));
    return true;
}

void SkeletonTracker::shutdown() {
    if (!initialized.load()) return;
    stopInfer = true;
    if (inferThread.joinable()) inferThread.join();
    delete ort;
    ort = nullptr;
    initialized = false;
    LOG("[SkeletonTracker] Shutdown complete");
}

void SkeletonTracker::setMaxPlayers(int n) {
    n = std::clamp(n, 1, kMaxPlayers);
    if (n == maxPlayers) return;
    maxPlayers = n;
    std::lock_guard<std::mutex> lock(resultMutex);
    results.resize(maxPlayers);
    pendingResults.resize(maxPlayers);
}

// ───────────────────────────────────────────────────────────────
// Frame submission (called from main/TD thread)
// ───────────────────────────────────────────────────────────────

void SkeletonTracker::submitFrame(const uint8_t* rgbaData, int width, int height) {
    if (!initialized.load() || !rgbaData) return;
    std::lock_guard<std::mutex> lock(frameMutex);
    size_t sz = static_cast<size_t>(width) * height * 4;
    submittedFrame.resize(sz);
    std::memcpy(submittedFrame.data(), rgbaData, sz);
    submittedWidth = width;
    submittedHeight = height;
    hasNewFrame = true;
}

int SkeletonTracker::getNumTrackedPersons() const {
    std::lock_guard<std::mutex> lock(resultMutex);
    int count = 0;
    for (const auto& p : results)
        if (p.active) count++;
    return count;
}

TrackedPerson SkeletonTracker::getPerson(int index) const {
    std::lock_guard<std::mutex> lock(resultMutex);
    if (index < 0 || index >= static_cast<int>(results.size())) return kEmptyPerson;
    return results[index];
}

// ───────────────────────────────────────────────────────────────
// Inference loop (background thread)
// ───────────────────────────────────────────────────────────────

void SkeletonTracker::inferLoop() {
    LOG("[SkeletonTracker] Inference thread started");

    while (!stopInfer.load()) {
        // Grab latest frame
        std::vector<uint8_t> localFrame;
        int w = 0, h = 0;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!hasNewFrame) {
                // Unlock before sleeping to avoid blocking submitFrame()
            } else {
                localFrame = submittedFrame;
                w = submittedWidth;
                h = submittedHeight;
                hasNewFrame = false;
            }
        }
        if (localFrame.empty() || w <= 0 || h <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        auto rgbNorm = rgbaToNormalizedRgb(localFrame.data(), w, h);

        // Run detector periodically or when no active detections
        bool runDetect = (frameCount % detectorInterval == 0) || lastDetections.empty();
        frameCount++;

        std::vector<Detection> detections;
        if (runDetect) {
            detections = runDetector(rgbNorm, w, h);
            // Cache bboxes
            lastDetections.clear();
            for (auto& d : detections) {
                lastDetections.push_back({d.cx, d.cy, d.w, d.h});
            }
        } else {
            // Reuse last detections
            for (auto& bb : lastDetections) {
                Detection d;
                d.cx = bb[0]; d.cy = bb[1]; d.w = bb[2]; d.h = bb[3];
                d.score = 0.5f;
                detections.push_back(d);
            }
        }

        // Limit to maxPlayers
        if (static_cast<int>(detections.size()) > maxPlayers) {
            std::partial_sort(detections.begin(), detections.begin() + maxPlayers, detections.end(),
                              [](const Detection& a, const Detection& b) { return a.score > b.score; });
            detections.resize(maxPlayers);
        }

        // Run landmark for each detection
        std::vector<TrackedPerson> newPersons(maxPlayers);
        for (int i = 0; i < static_cast<int>(detections.size()); i++) {
            float conf = 0.f;
            auto joints = runLandmark(rgbNorm, w, h, detections[i], conf);
            newPersons[i].joints = joints;
            newPersons[i].confidence = conf;
            newPersons[i].bbox[0] = detections[i].cx;
            newPersons[i].bbox[1] = detections[i].cy;
            newPersons[i].bbox[2] = detections[i].w;
            newPersons[i].bbox[3] = detections[i].h;
            newPersons[i].active = (conf > 0.3f);

            // Update cached bbox from landmark result (for tracking between detector runs)
            if (newPersons[i].active && i < static_cast<int>(lastDetections.size())) {
                // Recompute bbox from shoulder/hip landmarks
                float minX = 1.f, minY = 1.f, maxX = 0.f, maxY = 0.f;
                for (auto& j : joints) {
                    if (j.visibility > 0.3f) {
                        minX = std::min(minX, j.x);
                        minY = std::min(minY, j.y);
                        maxX = std::max(maxX, j.x);
                        maxY = std::max(maxY, j.y);
                    }
                }
                float pad = 0.1f;
                lastDetections[i] = {
                    (minX + maxX) * 0.5f,
                    (minY + maxY) * 0.5f,
                    (maxX - minX) + pad,
                    (maxY - minY) + pad
                };
            }
        }

        assignPersonIds(newPersons);

        // Publish results
        {
            std::lock_guard<std::mutex> lock(resultMutex);
            results = newPersons;
        }
    }

    LOG("[SkeletonTracker] Inference thread exiting");
}

// ───────────────────────────────────────────────────────────────
// SSD Anchor generation for BlazePose detector
// ───────────────────────────────────────────────────────────────

void SkeletonTracker::generateAnchors() {
    anchors.clear();
    const int strides[] = {8, 16, 32, 32, 32};
    const int numStrides = 5;

    for (int i = 0; i < numStrides; i++) {
        int stride = strides[i];
        int gridH = kDetectorInputSize / stride;
        int gridW = kDetectorInputSize / stride;
        // 2 anchors per grid cell
        for (int y = 0; y < gridH; y++) {
            for (int x = 0; x < gridW; x++) {
                float cx = (static_cast<float>(x) + 0.5f) / static_cast<float>(gridW);
                float cy = (static_cast<float>(y) + 0.5f) / static_cast<float>(gridH);
                anchors.push_back({cx, cy});
                anchors.push_back({cx, cy});
            }
        }
    }

    LOG("[SkeletonTracker] Generated " + std::to_string(anchors.size()) + " anchors");
}

// ───────────────────────────────────────────────────────────────
// Detector inference
// ───────────────────────────────────────────────────────────────

std::vector<SkeletonTracker::Detection>
SkeletonTracker::runDetector(const std::vector<float>& rgbNorm, int srcW, int srcH) {
    if (!ort || !ort->detectorSession) return {};

    auto resized = cropAndResize(rgbNorm, srcW, srcH,
                                  0.5f, 0.5f,
                                  1.0f, kDetectorInputSize);

    // NHWC layout: [1, 224, 224, 3]
    std::array<int64_t, 4> inputShape = {1, kDetectorInputSize, kDetectorInputSize, 3};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        ort->memInfo, resized.data(), resized.size(),
        inputShape.data(), inputShape.size());

    // Query model I/O names dynamically for portability across model variants
    Ort::AllocatorWithDefaultOptions allocator;
    auto inName = ort->detectorSession->GetInputNameAllocated(0, allocator);
    const char* inputNames[] = {inName.get()};

    size_t numOut = ort->detectorSession->GetOutputCount();
    std::vector<std::string> outNameStrs;
    outNameStrs.reserve(numOut);
    for (size_t i = 0; i < numOut; i++) {
        auto n = ort->detectorSession->GetOutputNameAllocated(i, allocator);
        outNameStrs.push_back(n.get());
    }
    std::vector<const char*> outNamePtrs;
    for (auto& s : outNameStrs) outNamePtrs.push_back(s.c_str());

    std::vector<Ort::Value> outputs;
    try {
        outputs = ort->detectorSession->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outNamePtrs.data(), outNamePtrs.size());
    } catch (const Ort::Exception& e) {
        LOG("[SkeletonTracker] Detector inference failed: " + std::string(e.what()));
        return {};
    }

    // Identify which output is regressors ([1,2254,12]) vs scores ([1,2254,1])
    const float* scores = nullptr;
    const float* regressors = nullptr;
    for (size_t i = 0; i < outputs.size(); i++) {
        auto info = outputs[i].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if (shape.size() == 3 && shape[2] == 12) {
            regressors = outputs[i].GetTensorData<float>();
        } else if (shape.size() == 3 && shape[2] == 1) {
            scores = outputs[i].GetTensorData<float>();
        }
    }
    if (!scores || !regressors) return {};

    std::vector<Detection> dets;
    const float scoreThreshold = 0.5f;

    for (int i = 0; i < kNumAnchors && i < static_cast<int>(anchors.size()); i++) {
        float score = 1.f / (1.f + std::exp(-scores[i])); // sigmoid
        if (score < scoreThreshold) continue;

        const float* reg = regressors + i * 12;
        float anchorCx = anchors[i][0];
        float anchorCy = anchors[i][1];

        Detection d;
        d.cx = anchorCx + reg[0] / static_cast<float>(kDetectorInputSize);
        d.cy = anchorCy + reg[1] / static_cast<float>(kDetectorInputSize);
        d.w  = reg[2] / static_cast<float>(kDetectorInputSize);
        d.h  = reg[3] / static_cast<float>(kDetectorInputSize);
        d.score = score;

        dets.push_back(d);
    }

    dets = nms(dets, 0.3f);
    return dets;
}

// ───────────────────────────────────────────────────────────────
// Landmark inference (per person)
// ───────────────────────────────────────────────────────────────

std::array<Joint, kNumJoints>
SkeletonTracker::runLandmark(const std::vector<float>& rgbNorm,
                              int srcW, int srcH,
                              const Detection& det,
                              float& outConfidence) {
    std::array<Joint, kNumJoints> joints{};
    outConfidence = 0.f;

    if (!ort || !ort->landmarkSession) return joints;

    // Crop region: square around detection bbox with padding
    float cropSize = std::max(det.w, det.h) * 1.25f;
    auto cropped = cropAndResize(rgbNorm, srcW, srcH,
                                  det.cx, det.cy, cropSize,
                                  kLandmarkInputSize);

    // NHWC: [1, 256, 256, 3]
    std::array<int64_t, 4> inputShape = {1, kLandmarkInputSize, kLandmarkInputSize, 3};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        ort->memInfo, cropped.data(), cropped.size(),
        inputShape.data(), inputShape.size());

    // The Unity BlazePose landmark model may have different output names.
    // We'll query them dynamically.
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numOutputs = ort->landmarkSession->GetOutputCount();
    std::vector<std::string> outputNameStrs;
    outputNameStrs.reserve(numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
        auto namePtr = ort->landmarkSession->GetOutputNameAllocated(i, allocator);
        outputNameStrs.push_back(namePtr.get());
    }
    std::vector<const char*> outputNamePtrs;
    for (auto& s : outputNameStrs) outputNamePtrs.push_back(s.c_str());

    size_t numInputs = ort->landmarkSession->GetInputCount();
    std::vector<std::string> inputNameStrs;
    inputNameStrs.reserve(numInputs);
    for (size_t i = 0; i < numInputs; i++) {
        auto namePtr = ort->landmarkSession->GetInputNameAllocated(i, allocator);
        inputNameStrs.push_back(namePtr.get());
    }
    std::vector<const char*> inputNamePtrs;
    for (auto& s : inputNameStrs) inputNamePtrs.push_back(s.c_str());

    std::vector<Ort::Value> outputs;
    try {
        outputs = ort->landmarkSession->Run(
            Ort::RunOptions{nullptr},
            inputNamePtrs.data(), &inputTensor, 1,
            outputNamePtrs.data(), outputNamePtrs.size());
    } catch (const Ort::Exception& e) {
        LOG("[SkeletonTracker] Landmark inference failed: " + std::string(e.what()));
        return joints;
    }

    // Identify outputs by shape:
    //   [1, 195] = landmarks (39 landmarks x 5 values; first 33 are pose)
    //   [1, 1]   = pose confidence flag
    //   [1, 117] = world landmarks (39 x 3)
    //   [1, 256, 256, 1] = segmentation mask (ignored)
    //   [1, 64, 64, 39]  = heatmap (ignored)
    const float* landmarkData = nullptr;
    size_t landmarkSize = 0;
    float poseFlag = 0.f;

    for (size_t i = 0; i < outputs.size(); i++) {
        auto info = outputs[i].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        size_t totalElements = info.GetElementCount();

        if (shape.size() == 2 && totalElements >= 195) {
            landmarkData = outputs[i].GetTensorData<float>();
            landmarkSize = totalElements;
        } else if (shape.size() == 2 && totalElements == 1) {
            poseFlag = 1.f / (1.f + std::exp(-outputs[i].GetTensorData<float>()[0]));
        }
    }

    if (!landmarkData) return joints;

    outConfidence = poseFlag;

    // 195 = 39 landmarks x 5 values (x, y, z, visibility, presence)
    // We only need the first 33 pose landmarks; remaining 6 are auxiliary
    constexpr int kValuesPerLandmark = 5;
    int valuesPerLandmark = kValuesPerLandmark;

    float cropLeft = det.cx - cropSize * 0.5f;
    float cropTop  = det.cy - cropSize * 0.5f;

    for (int j = 0; j < kNumJoints; j++) {
        int base = j * valuesPerLandmark;
        float lx = landmarkData[base + 0] / static_cast<float>(kLandmarkInputSize);
        float ly = landmarkData[base + 1] / static_cast<float>(kLandmarkInputSize);
        float lz = (valuesPerLandmark >= 3) ? landmarkData[base + 2] : 0.f;
        float vis = (valuesPerLandmark >= 4) ?
                    1.f / (1.f + std::exp(-landmarkData[base + 3])) : 1.f;

        // Transform from crop space to original image normalized coords
        joints[j].x = cropLeft + lx * cropSize;
        joints[j].y = cropTop  + ly * cropSize;
        joints[j].z = lz / static_cast<float>(kLandmarkInputSize);
        joints[j].visibility = vis;
    }

    return joints;
}

// ───────────────────────────────────────────────────────────────
// Image preprocessing
// ───────────────────────────────────────────────────────────────

std::vector<float> SkeletonTracker::rgbaToNormalizedRgb(const uint8_t* rgba, int w, int h) {
    std::vector<float> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        rgb[i * 3 + 0] = static_cast<float>(rgba[i * 4 + 0]) / 255.f;
        rgb[i * 3 + 1] = static_cast<float>(rgba[i * 4 + 1]) / 255.f;
        rgb[i * 3 + 2] = static_cast<float>(rgba[i * 4 + 2]) / 255.f;
    }
    return rgb;
}

std::vector<float> SkeletonTracker::cropAndResize(const std::vector<float>& rgb,
                                                   int srcW, int srcH,
                                                   float cx, float cy, float size,
                                                   int dstSize) {
    std::vector<float> out(dstSize * dstSize * 3, 0.f);

    float left = cx - size * 0.5f;
    float top  = cy - size * 0.5f;

    for (int dy = 0; dy < dstSize; dy++) {
        for (int dx = 0; dx < dstSize; dx++) {
            float srcXf = left * srcW + (static_cast<float>(dx) / dstSize) * size * srcW;
            float srcYf = top  * srcH + (static_cast<float>(dy) / dstSize) * size * srcH;

            int sx = static_cast<int>(srcXf);
            int sy = static_cast<int>(srcYf);

            if (sx < 0 || sx >= srcW || sy < 0 || sy >= srcH) continue;

            int srcIdx = (sy * srcW + sx) * 3;
            int dstIdx = (dy * dstSize + dx) * 3;
            out[dstIdx + 0] = rgb[srcIdx + 0];
            out[dstIdx + 1] = rgb[srcIdx + 1];
            out[dstIdx + 2] = rgb[srcIdx + 2];
        }
    }
    return out;
}

// ───────────────────────────────────────────────────────────────
// Non-Maximum Suppression
// ───────────────────────────────────────────────────────────────

float SkeletonTracker::computeIoU(const Detection& a, const Detection& b) {
    float ax1 = a.cx - a.w * 0.5f, ay1 = a.cy - a.h * 0.5f;
    float ax2 = a.cx + a.w * 0.5f, ay2 = a.cy + a.h * 0.5f;
    float bx1 = b.cx - b.w * 0.5f, by1 = b.cy - b.h * 0.5f;
    float bx2 = b.cx + b.w * 0.5f, by2 = b.cy + b.h * 0.5f;

    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float areaA = a.w * a.h, areaB = b.w * b.h;
    return inter / (areaA + areaB - inter + 1e-6f);
}

std::vector<SkeletonTracker::Detection>
SkeletonTracker::nms(std::vector<Detection>& dets, float iouThresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (!suppressed[j] && computeIoU(dets[i], dets[j]) > iouThresh) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

// ───────────────────────────────────────────────────────────────
// Person ID tracking across frames (IoU-based)
// ───────────────────────────────────────────────────────────────

void SkeletonTracker::assignPersonIds(std::vector<TrackedPerson>& persons) {
    std::lock_guard<std::mutex> lock(resultMutex);

    // Match new detections to existing results by IoU
    std::vector<bool> matchedOld(results.size(), false);
    std::vector<bool> matchedNew(persons.size(), false);

    for (size_t ni = 0; ni < persons.size(); ni++) {
        if (!persons[ni].active) continue;

        float bestIoU = 0.3f;
        int bestOld = -1;

        for (size_t oi = 0; oi < results.size(); oi++) {
            if (matchedOld[oi] || !results[oi].active) continue;

            Detection dNew, dOld;
            dNew.cx = persons[ni].bbox[0]; dNew.cy = persons[ni].bbox[1];
            dNew.w  = persons[ni].bbox[2]; dNew.h  = persons[ni].bbox[3];
            dOld.cx = results[oi].bbox[0]; dOld.cy = results[oi].bbox[1];
            dOld.w  = results[oi].bbox[2]; dOld.h  = results[oi].bbox[3];

            float iou = computeIoU(dNew, dOld);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestOld = static_cast<int>(oi);
            }
        }

        if (bestOld >= 0) {
            persons[ni].id = results[bestOld].id;
            matchedOld[bestOld] = true;
            matchedNew[ni] = true;
        }
    }

    // Assign new IDs to unmatched persons
    for (size_t ni = 0; ni < persons.size(); ni++) {
        if (!persons[ni].active) continue;
        if (!matchedNew[ni]) {
            persons[ni].id = nextPersonId++;
        }
    }
}
