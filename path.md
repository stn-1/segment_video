Tổng hợp lại toàn bộ điều kiện của bạn:
✓ Video du lịch, 5-10 phút
✓ Mục tiêu: timestamp + embedding + caption
✓ Ưu tiên tốc độ
✓ False merge chấp nhận được hơn false cut
✓ Không cần recall tuyệt đối

Kết luận rõ ràng
2 stage là lựa chọn tối ưu nhất cho bạn, không cần bàn thêm.
Stage 1 — PySceneDetect AdaptiveDetector
threshold = 4.0 ← tránh cut do người đi qua
min_scene_len = 60 ← tối thiểu 2 giây/segment

Stage 2 — DINOv2-ViT-S verify
sample 5 keyframe/segment
cosine_sim > 0.82 → merge

Lý do không chọn các hướng khác
Sliding window → thêm 18s, không cải thiện caption ❌
Temporal clustering → phức tạp, khó tune, chậm hơn ❌
Chỉ PySceneDetect → vẫn bị false cut do chuyển động ❌
Chỉ embedding → quá chậm, không cần thiết ❌

Timeline thực tế pipeline hoàn chỉnh
Video 10 phút
│
├── PySceneDetect ~5s ← nhanh, CPU
├── DINOv2 verify ~4s ← nhẹ, GPU  
 ├── Chọn keyframe ~1s ← frame gần mean embedding nhất
└── InternVL2-2B ~50s ← bottleneck thật sự

Tổng: ~60s cho video 10 phút ✓
Thời gian còn lại nên tập trung optimize InternVL2-2B — batch inference, quantization INT4 — đó mới là nơi tiết kiệm được nhiều nhất.
