# Sampler replay fraction/cap/luân phiên — Prompt 8

Đã thêm API và tests synthetic. **Chưa nối trainer/config, chưa train dataset thật.**
Không thay `FixedReplaySampler` legacy hoặc effective batch/loss reduction.

File mới:

- `src/data/fold_replay_sampler.py`: `FoldReplayFractionSampler`, policy
  `fold_fraction_capped_rotating_v1`.
- `tests/test_fold_replay_sampler.py`: current coverage, rounding/cap/fairness,
  A/B/RNG independence, state round-trip, guards và DataLoader tail.

## 1. Số lượt học mỗi epoch

Với `N` current samples, `M` exemplar đang có và mặc định `f=0.125`, `cap=3`:

```text
target_replay = floor(N * f / (1-f)) = floor(N/7)
actual_replay = min(target_replay, 3*M)
total_draws = N + actual_replay
actual_fraction = actual_replay / total_draws
```

Fraction tính bằng rational từ chuỗi số, rounding luôn **floor**, không stochastic
rounding. Do rounding hoặc thiếu capacity, fraction thực tế có thể dưới 12,5%.
Task 0 replay bằng 0; truyền memory không rỗng ở task 0 sẽ fail để tránh dùng nhầm
state cũ. Current rỗng được primitive hỗ trợ (0 draws), nhưng trainer sau này phải
kiểm tra một task train rỗng có hợp lệ hay không.

Ví dụ:

- `N=149000, M=9259`: target 21.285, capacity 27.777 → dùng 21.285 replay draws.
- `N=140, M=3`: target 20, capacity 9 → chỉ replay 9; vẫn giữ đủ 140 current draws.
- `N=6`: target 0, không ép thêm một replay draw để đạt tỷ lệ gần đúng.
- Buffer lớn không có nghĩa replay hết buffer mỗi epoch.

Mỗi current ID xuất hiện **đúng một lần**. Sampler xuất index cho dataset nối
`[current_rows, replay_rows]`, không chia microbatch và không drop dòng cuối.
DataLoader phải dùng `drop_last=False`; không yêu cầu mỗi batch đúng 56 current + 8
replay. Cấu hình batch64/effective1024/accum16 và loss reduction thuộc bước trainer,
không được sửa ở bước này.

## 2. Fairness giữa class và exemplar

1. Tạo thứ tự class cố định, có seed riêng theo member/task.
2. Cấp từng lượt theo round-robin; class chạm `cap × số exemplar` được bỏ khỏi
   hàng đợi **trong epoch đó**, lượt dư chuyển cho class còn khả năng replay.
3. Epoch sau reset repeat budget; class cursor tiếp tục từ vị trí sau class cuối
   đã được chọn. Nếu chỉ có một replay draw mỗi epoch, các class được luân phiên.
4. Mỗi class có queue exemplar riêng: sort IDs trước, rồi permutation bằng RNG
   member/task/class. Queue cố định trong task, cursor tiếp tục qua các epoch.
5. Đi hết queue rồi mới lặp. Không chọn đi chọn lại rank prefix đầu buffer.
   Vì mỗi class nhận tối đa `cap*M_class` lượt, một ID không vượt cap trong epoch.
6. Current được shuffle; replay được chèn vào các vị trí ngẫu nhiên nhưng **giữ
   thứ tự tương đối của replay**. Không shuffle toàn bộ index sau cùng, vì cách đó
   có thể đưa một lần lặp lên trước khi các exemplar khác trong cycle được dùng.

Ví dụ rare class có 1 exemplar, class khác có 10: target 20 lượt → rare class tối
đa 3, class còn lại nhận 17. Repeat counters reset mỗi epoch, cursor không reset.

## 3. RNG và hai pilot A/B

Derivation ghi rõ: `seedsequence_pcg64_member_task_epoch_stream_v1`.
Member seed lấy từ derivation hiện có trong `src/utils/seed.py`.

- Generator NumPy **PCG64 riêng**, không gọi `set_global_seed`, không dùng Python
  `hash()` hoặc tiêu thụ Python/NumPy/Torch RNG chung.
- Stream riêng cho thứ tự class, queue exemplar, current shuffle và vị trí trộn.
- Current shuffle/mixing dùng member/task/local epoch. Queue dùng member/task/class.
- Một task mới tạo **sampler instance mới**, local epoch 0 và cursor ban đầu mới.
  Không truyền cursor của task cũ sang task mới.
- A dừng task0 ở epoch3 và B dừng epoch13: tại cùng task1/local epoch, nếu IDs/labels
  current và memory giống nhau thì sample order vẫn giống nhau.
- Canonical IDs trước RNG làm sample-ID sequence không phụ thuộc vị trí dòng input.
  Tuy vậy resume vẫn kiểm tra **ordered IDs/labels hashes**, không dùng lại index
  state với dataset đã đổi thứ tự mà không phát hiện.

Sampler chỉ biết IDs/labels đã được cung cấp, không đọc dataset/folds/task-file.
Kiểm tra held-out/test/future thuộc loader/buffer và tích hợp Prompt 9; caller phải
truyền current của task đang học và memory từ các task đã hoàn thành, trước khi
buffer update với current task.

## 4. State, preview và resume

- `plan_epoch(e)`: preview thuần, trả `ReplayEpochPlan(indices, audit)`, không đổi
  trạng thái. Có thể dùng cho kiểm tra A/B/dry-run sampler.
- `set_epoch(e)`: tái dựng cursor từ đầu task đến local epoch `e` một cách
  deterministic; không phụ thuộc số epoch từng chạy ở task trước.
- `iter(sampler)`: sinh index; khi iterator đã exhausted thì chuyển `next_epoch`
  và cập nhật `last_audit`. Không cho hai iterator active cùng lúc.
- `state_dict()`: JSON-safe, gồm schema/config/IDs hashes, next epoch, class order,
  class cursor, exemplar queues/cursors, RNG derivation/state metadata và checksum.
- RNG là keyed/stateless nên không có mutable PRNG state phải đoán: artifact ghi
  `mutable_rng_state=null`, epoch key và queue đã suy ra.
- `from_state_dict(state, **constructor_arguments)`: kiểm tra checksum, member/
  seed/task/fraction/cap/ID-label-order và tái dựng đối chiếu queues/cursors.

**Chỉ hỗ trợ sampler state ở epoch boundary**, không resume giữa epoch. Khi iterator
còn active, `state_dict` và `set_epoch` fail. Nếu chủ động bỏ iterator, gọi `close()`
để không commit epoch đó; lần gọi lại sẽ có cùng order. Việc này **không rollback
model/optimizer** đã train một phần epoch, nên không dùng nó thay checkpoint training.

DataLoader có thể prefetch trước khi model xử lí xong: iterator exhausted chỉ chứng
minh indices đã phát hết, không chứng minh optimizer đã hoàn tất epoch. Trainer phải
lưu checkpoint sau vòng train hoàn tất. Checkpoint task boundary đầy đủ vẫn thuộc
Prompt 10, không được tự suy từ sampler state.

## 5. Audit

`last_audit` chỉ giữ epoch đã được iterator phát hết, không giữ lịch sử lớn trong RAM.
Trainer sau này chịu trách nhiệm ghi từng audit ra file.

- Target/actual replay draws, total/current draws, target/actual fraction và
  `capacity_limited`.
- Per-class draws, unique exemplars, memory size, coverage và max repeat.
- Tổng unique replay exemplars, class/exemplar coverage, repeat histogram (gồm
  các ID được dùng 0 lần), max repeat và cap.
- Hash thứ tự replay IDs và hash thứ tự toàn bộ sample IDs được phát.

Coverage lấy mẫu số là **class/exemplar đang có trong buffer**. Class cũ mất hết
exemplar không thể được sampler suy ra từ memory rỗng; trainer cần báo riêng số
old seen classes không còn exemplar. Không gọi coverage này là coverage của toàn
bộ old dataset/178 classes.

## 6. API minh họa — không train tự động

```python
from src.data.fold_replay_sampler import FoldReplayFractionSampler

sampler = FoldReplayFractionSampler(
    current_sample_ids=current.metadata["sample_id"],
    replay_sample_ids=memory.metadata["sample_id"],
    replay_raw_labels=memory.labels,
    experiment_seed=0, member_id=0, task_id=1,
    replay_fraction=0.125, repeat_cap=3,
)
plan = sampler.plan_epoch(0)  # không advance state

# Trainer sau này dùng dataset nối current + memory theo đúng thứ tự input.
# loader = DataLoader(dataset, sampler=sampler, batch_size=64, drop_last=False)
# ... train hoàn tất epoch ...
# audit = sampler.last_audit
# state = sampler.state_dict()
```

Test synthetic:

```bash
python -m pytest tests/test_fold_replay_sampler.py tests/test_fold_replay_buffer.py tests/test_s04_fixed_budget.py tests/test_replay_checkpoint.py -q
```

Tiếp theo: **Prompt 9 — tích hợp trainer**, chưa tự chạy pilot/full experiment.
