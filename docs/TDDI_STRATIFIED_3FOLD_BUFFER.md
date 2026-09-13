# Buffer quota sqrt và ranking cho pilot A/B — Prompt 7

Trạng thái: API và tests synthetic đã có. **Chưa nối sampler/trainer/config,
chưa chạy dataset thật.** Không sửa `FixedBudgetReplayBuffer`, allocator uniform,
`FixedReplaySampler` hoặc checkpoint legacy.

Theo [decision record](TDDI_STRATIFIED_3FOLD_DECISION_RECORD.md), ranking dưới đây
chỉ là **control tạm thời cho hai pilot preprocessing**, không khẳng định là cách
chọn exemplar tốt nhất hoặc toàn bộ recipe paper.

## 1. File và phạm vi

- `src/data/fold_replay_buffer.py`: module mới opt-in, không thay buffer cũ.
  - `four_percent_member_budgets`: tính tổng 4% rồi chia ba member.
  - `min_quota_sqrt_allocation`: quota nền + sqrt weights với capacity thực tế.
  - `rank_raw_class_exemplars`: chọn gần mean class trong không gian normalize
    từng mẫu, không dùng scaler/model/UE.
  - `FoldSqrtReplayBuffer`: kiểm tra provenance, cập nhật theo task, giữ raw
    exemplars và cung cấp state để bước checkpoint sau tích hợp.
  - `ensemble_buffer_accounting`: báo slots, unique IDs và overlap giữa ba buffer.
- `tests/test_fold_replay_buffer.py`: allocation/ranking tính tay, leakage guards,
  prefix trimming, A/B equality, global accounting và state round-trip.

Policy được ghi trong metadata/audit là `fold_min_quota_sqrt_capacity_v1`, khác
policy uniform cũ. Không có lệnh train mới trong bước này.

## 2. Budget và quota

Với 694.455 development rows:

```text
global slots = floor(4 * 694455 / 100) = 27778
member 0 = 9260
member 1 = 9259
member 2 = 9259
```

Phép tính dùng integer, không làm tròn float. Cho phép overlap giữa member,
nhưng mỗi bản lưu vẫn tính một slot. Một ID lưu ở hai member = hai slots, không
phải một. Helper accounting kiểm tra đúng ba member, cùng stage/protocol/seed và
tổng budget; không cấp 27.778 slots riêng cho mỗi member.

Allocator nhận hai thông tin **khác nhau** cho mỗi class đã xuất hiện:

| Thông tin | Ý nghĩa |
| --- | --- |
| `observed_counts` | Số training samples của member khi class xuất hiện; giữ cố định làm sqrt weight |
| `feasible_capacities` | Class mới: current samples; class cũ: chỉ exemplar đang giữ |

Không dùng tần suất validation/test/future classes để chia slots. Buffer chỉ nhận
đầy đủ current rows thuộc task hiện tại và hai training folds; dữ liệu thiếu,
trùng, nhầm fold/class/source đều fail. Task file có thể chứa future class IDs,
nhưng không lấy count của chúng làm trọng số trước khi chúng xuất hiện.

Quy tắc integer deterministic:

1. Target = `min(budget, tổng feasible capacities)`.
2. Quota nền mỗi class = `min(10, capacity)`.
3. Nếu target không đủ quota nền: chia max-min đều trong các quota nền đã clip;
   lượt dư ưu tiên raw class ID nhỏ hơn. Bước này tái sử dụng helper max-min cũ,
   không thay behavior của helper.
4. Nếu đủ nền: phần dư chia theo `sqrt(observed_count)`.
5. Class chạm capacity nhận tối đa khả năng giữ, rồi phân phối lại phần còn dư.
6. Khi không còn saturation: lấy floor các shares; chia slots dư theo phần thập
   phân lớn nhất (*largest remainder*), hòa thì raw class ID nhỏ hơn.

Ví dụ observed 100/400, capacity đủ lớn, budget 50: nền 10/10; dư 30 chia tỷ lệ
10:20 → kết quả **20/30**. Class hiếm chỉ có 4 mẫu thì tối đa giữ 4, không nhân bản.

Nếu class cũ từng thấy 100 mẫu nhưng chỉ còn 2 exemplar, nó có **weight sqrt(100)**
nhưng **capacity 2**. Không lấy lại 98 mẫu đã bỏ khi quota lý tưởng muốn tăng.
Class đã mất hết exemplar có capacity 0. Budget nhỏ/0 và tổng feasible nhỏ hơn
budget đều có hành vi được test; không bắt buộc buffer phải đầy khi không đủ mẫu.

## 3. Ranking và dữ liệu lưu

Policy: `raw_sample_normalized_class_mean_control_v1`.

```text
z_i = (raw_i - mean(raw_i)) / sqrt(var(raw_i, ddof=0) + 1e-5)
class_center = mean(z_i trong class)
distance_i = Euclidean(z_i, class_center)
```

- Float64 cho normalization, mean, distance và raw storage.
- Không learned affine; không dùng input scaler A/B, model latent hoặc UE.
- Sắp xếp sample IDs trước khi tính mean class để input shuffle không đổi thứ tự
  phép cộng; sort kết quả theo distance tăng, hòa theo sample ID lexicographic.
- Mẫu có mọi descriptor bằng nhau normalize thành vector 0.
- NaN/Inf/overflow ở input hoặc bước trung gian đều fail; không tự impute/clip/drop.
- Chọn prefix gần mean. Class cũ chỉ trim prefix đã xếp hạng lúc xuất hiện;
  không tính lại center/ranking bằng tập nhỏ đang giữ.

Buffer lưu **raw descriptors của các mẫu còn giữ**, raw labels, sample IDs,
source split/index, fold, drug IDs, rank priority và distance lúc class xuất hiện.
Source hashes nằm ở metadata chung, nên có thể đổi physical path khi bytes giữ
nguyên. Slice cũ được copy để không giữ backing array chứa mẫu đã bỏ.

Không giữ normalization vectors hoặc full old feature matrix. Context dùng chung
chỉ chứa identity/label/fold metadata phục vụ integrity, không chứa descriptors.
Audit history chỉ có counts/capacities/allocations, không cất raw features đã bỏ.

`get_all()` trả **raw** arrays theo raw class ID rồi rank, kèm metadata và defensive
copies. `model_arrays(preprocessing)` áp artifact Prompt 6 cho raw đang giữ: A/B
nhận cùng IDs/rank nhưng model inputs khác. Buffer không biết lựa chọn preprocessing
nào thắng; transform này không làm đổi raw storage/ranking.

Caller bắt buộc đưa raw từ loader vào `update`, không đưa features đã scale. Buffer
kiểm tra ID/label/source/fold/feature order nhưng không thể suy ra một matrix tùy ý
đã được scale hay chưa chỉ từ values. Prompt trainer sau phải giữ đúng đường raw.

## 4. API minh họa — chưa phải lệnh training

```python
from src.data.ddi_dataset import load_development_fold_arrays
from src.data.fold_replay_buffer import FoldSqrtReplayBuffer, four_percent_member_budgets

# context phải vừa validate ở đầu run/resume theo Prompt 5.
budgets = four_percent_member_budgets(694455)
buffer = FoldSqrtReplayBuffer(
    context=context,
    task_file="study_assets/task_protocols/tail_to_head_tasks.json",
    feature_columns=feature_columns,
    member_id=0, experiment_seed=0,
    total_memory_budget=budgets[0], base_quota=10,
)

raw_current = load_development_fold_arrays(
    context, feature_columns, role="train", member_id=0, validation_fold=0,
    class_ids=task0_classes,  # lấy đúng task-file, không dùng toàn bộ seen/future
)
audit = buffer.update(raw_current, task_id=0, feature_columns=feature_columns)
raw_memory = buffer.get_all()
model_memory = buffer.model_arrays(frozen_preprocessing)
```

Đây là ví dụ gọi API chọn memory tại boundary. Sau này trainer sẽ update memory
sau khi học task hiện tại; **không** dùng ví dụ này để đưa current vào replay trước
khi học. `update` yêu cầu task 0→1→2..., mỗi task đúng một lần; không nhận class cũ
đã loại bỏ để bổ sung buffer.

## 5. State và audit cho bước resume sau

`state_dict()` trả cấu trúc độc lập với arrays đang sống:

- Schema/policy/ranking conventions, budget/quota, seed derivation, hashes và
  feature order.
- Completed task ID, observed arrival counts và audit từng task.
- Chỉ retained raw entries + labels/IDs/source/rank.
- SHA256 nội dung state (numeric arrays dùng bytes little-endian).

`FoldSqrtReplayBuffer.from_state_dict(state, **constructor_arguments)` yêu cầu
context mới validate đầu resume. Kiểm tra checksum, metadata/member/budget/task/
source/feature order; đối chiếu exact provenance, count/capacity/allocation history
và rank prefix. Không fit scaler, không rerank, không đọc lại old feature Parquet.

State API này **chưa phải checkpoint trainer hoàn chỉnh**, không sửa checkpoint
EWC/replay cũ và không tự ghi file. File-level atomic checkpoint sẽ thuộc Prompt 10.
Không tự load pickle/checkpoint từ nguồn không tin cậy.

Audit gồm `policy`, `task_id`, `observed_counts`, `feasible_capacities`, `allocation`,
`stored_slots`, `budget`. Model-input policy/hash vẫn do preprocessing artifact
quản lý; không đưa nó vào thuật toán chọn IDs để giữ A/B đồng nhất.

## 6. Kiểm thử và bước tiếp theo

```bash
python -m pytest tests/test_fold_replay_buffer.py tests/test_s04_fixed_budget.py tests/test_replay_checkpoint.py tests/test_fold_preprocessing.py tests/test_development_fold_loader.py -q
```

Toàn bộ dùng synthetic data, không chạy dataset thật. Các tests cover tính tay,
saturation/redistribution, budget nhỏ, missing/duplicate/held-out/test/future rows,
old prefix/no reread, A/B, source relocation, state round-trip và legacy.

Tiếp theo **Prompt 8 — sampler fraction/cap/luân phiên**. Chưa đổi sampler hiện tại
thành 12,5%/cap 3 chỉ bằng việc thêm buffer mới.
