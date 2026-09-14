# MIND Research

Project nghiên cứu hệ thống gợi ý tin tức dựa trên bộ dữ liệu [Microsoft News Dataset (MIND)](https://msnews.github.io/). Project kết hợp:

- **Semantic representation** bằng Sentence-BERT để biểu diễn ngữ nghĩa của bài viết.
- **Topic representation** bằng BERTopic để biểu diễn chủ đề.
- **User Representation Vector (URV)** để tổng hợp lịch sử đọc của người dùng.
- **Recommendation engine** để xếp hạng các bài viết ứng viên theo độ phù hợp.
- **Demo web/API** được đóng gói thành các Docker service.

## Cấu trúc project

```text
data/          Script tải dữ liệu và các tập MIND đã tải về.
demo/          Demo chạy bằng FastAPI và Docker Compose.
models/        Model BERTopic đã train/lưu trữ.
notebooks/     Notebook dùng cho phân tích dữ liệu, train và đánh giá model.
results/       Kết quả dự đoán và đánh giá.
src/           Code chính của pipeline biểu diễn và gợi ý.
vectors/       Vector biểu diễn được sinh từ dữ liệu.
```

### Các thư mục chính

- `data/download.py`: tải và giải nén các file `behaviors.tsv` và `news.tsv` của MIND-large.
- `data/primary/`: dữ liệu MIND theo các split `train_set`, `dev_set`, `test_set`.
- `src/core/`: model dữ liệu và các cấu trúc vector dùng chung.
- `src/semantic/`: tạo và huấn luyện biểu diễn semantic.
- `src/topic/`: tạo và huấn luyện biểu diễn topic bằng BERTopic.
- `src/represent/`: kết hợp các biểu diễn thành vector bài viết.
- `src/urv/`: tạo vector đại diện cho sở thích người dùng từ lịch sử đọc.
- `src/recommendation/`: sinh ứng viên và xếp hạng tin tức.
- `src/matrix/`: tạo ma trận dữ liệu phục vụ phân tích/đánh giá.
- `src/evaluater/`: đánh giá kết quả recommendation.
- `demo/semantic/`: service mã hóa semantic, mặc định cổng `8001`.
- `demo/topic/`: service mã hóa topic, mặc định cổng `8002`.
- `demo/recommend/`: service recommendation và API phục vụ demo web, mặc định cổng `8003`.
- `demo/web/`: giao diện web của demo.
- `notebooks/`: các bước thực nghiệm từ phân tích dữ liệu đến tạo prediction.

## Cài đặt môi trường

Yêu cầu Python 3.11 trở lên. Tạo virtual environment và cài dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Tải dữ liệu MIND

Script tải dữ liệu từ Hugging Face repository `yjw1029/MIND`, sau đó chỉ giữ lại hai file cần thiết là `behaviors.tsv` và `news.tsv`.

Chạy từ thư mục gốc project:

```bash
python data/download.py
```

Sau khi chạy xong, dữ liệu nằm tại:

```text
data/primary/
├── train_set/behaviors.tsv
├── train_set/news.tsv
├── dev_set/behaviors.tsv
├── dev_set/news.tsv
├── test_set/behaviors.tsv
└── test_set/news.tsv
```

## Chạy demo bằng Docker

Yêu cầu Docker và Docker Compose. Demo gồm ba service:

| Service | Cổng | Chức năng |
| --- | ---: | --- |
| `semantic` | `8001` | Mã hóa semantic bằng Sentence-BERT |
| `topic` | `8002` | Mã hóa topic bằng BERTopic |
| `recommend` | `8003` | API recommendation và giao diện web |

Trước khi chạy service `recommend`, cần chuẩn bị các artifact mà `demo/docker-compose.yml` mount vào container:

```text
data/raw/
vectors/mind_small/
models/mind_small/bertopic.pkl
demo/mind_verification.json
```

Các artifact này là dữ liệu/vector/model đã được tạo từ pipeline trong `src/` và các notebook. Nếu artifact được lưu ở vị trí khác, cập nhật volume và biến môi trường tương ứng trong `demo/docker-compose.yml`.

Build và khởi động toàn bộ demo:

```bash
docker compose -f demo/docker-compose.yml up --build
```

Mở giao diện tại [http://localhost:8003](http://localhost:8003). API documentation của service recommendation có tại [http://localhost:8003/docs](http://localhost:8003/docs).

Kiểm tra health của từng service:

```text
http://localhost:8001/semantic-health
http://localhost:8002/topic-health
http://localhost:8003/recommender-health
```

Dừng demo:

```bash
docker compose -f demo/docker-compose.yml down
```

## API chính

- `POST /semantic-encode`: tạo semantic vector cho danh sách text.
- `POST /topic-encode`: tạo topic vector từ text và semantic vector.
- `POST /recommender-recommend`: xếp hạng danh sách tin dựa trên lịch sử đọc.
- `GET /mind/metadata`: thông tin dataset và artifact đang dùng.
- `GET /mind/impressions`: lấy danh sách lượt đọc MIND.
- `POST /mind/impressions/{impression_id}/recommend`: xếp hạng ứng viên của một lượt đọc.
- `GET /mind/news`: tìm kiếm kho tin MIND.

## Ghi chú

- Dữ liệu MIND và các file model/vector có thể lớn nên được loại khỏi Git theo `.gitignore`.
- Lần chạy đầu tiên có thể mất thời gian để tải model Sentence-BERT và build dependency `hdbscan`.
- Demo sử dụng vector đã lưu để xếp hạng dữ liệu MIND; đây là demo nghiên cứu, không phải hệ thống tin tức thời gian thực.
