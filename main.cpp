#include <sys/mman.h>
#include <iostream>
#include <fstream>
#include <random>
#include <climits>
#include <boost/program_options.hpp>
#include <unistd.h>

#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/schema.h"
#include "arrow/util/status.h"
#include "arrow/util/memory-pool.h"
#include "arrow/util/buffer.h"
#include "arrow/types/primitive.h"
#include "arrow/io/memory.h"
#include "arrow/io/interfaces.h"
#include "arrow/ipc/adapter.h"

#include "immintrin.h"
#include "emmintrin.h"

using namespace arrow;
using namespace ipc;
using namespace io;
namespace po = boost::program_options;

const std::string data_name = "test.dat";
const std::string meta_name = "test.meta";

const auto INT32 = std::make_shared<Int32Type>();
const auto UINT32 = std::make_shared<UInt32Type>();
const auto UINT64 = std::make_shared<UInt64Type>();
const auto DOUBLE = std::make_shared<DoubleType>();
const auto FLOAT = std::make_shared<FloatType>();
const auto STRING = std::make_shared<StringType>();

// TPC-H schema
//auto f0 = std::make_shared<Field>("f0", UINT64);
//auto f1 = std::make_shared<Field>("f1", UINT32);
//auto f2 = std::make_shared<Field>("f2", FLOAT);
//auto f3 = std::make_shared<Field>("f3", UINT64);
//auto f4 = std::make_shared<Field>("f4", DOUBLE);
//auto f5 = std::make_shared<Field>("f5", UINT32);
//auto f6 = std::make_shared<Field>("f6", FLOAT);
//auto f7 = std::make_shared<Field>("f7", UINT64);
//auto f8 = std::make_shared<Field>("f8", DOUBLE);
//auto f9 = std::make_shared<Field>("f9", UINT32);
//auto f10 = std::make_shared<Field>("f10", FLOAT);
//auto f11 = std::make_shared<Field>("f11", UINT64);
//auto f12 = std::make_shared<Field>("f12", DOUBLE);
//auto f13 = std::make_shared<Field>("f13", STRING);
//auto f14 = std::make_shared<Field>("f14", STRING);
//auto f15 = std::make_shared<Field>("f15", STRING);
//auto f16 = std::make_shared<Field>("f16", STRING);

auto f0 = std::make_shared<Field>("f0", INT32);
auto f1 = std::make_shared<Field>("f1", FLOAT);

//std::shared_ptr<Schema> schema(new Schema({f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16}));
std::shared_ptr<Schema> schema(new Schema({f0, f1}));

struct MMapFile {
  FILE* file;
  uint8_t* mm;
  int64_t file_size;

  ~MMapFile() {
    munmap(mm, file_size);
    fclose(file);
  }
};

auto read_file(std::string file_name) -> std::shared_ptr<MMapFile> {
  std::shared_ptr<MMapFile> mm_file = std::make_shared<MMapFile>();

  mm_file->file = fopen(file_name.c_str(), "rb");
  fseek(mm_file->file, 0L, SEEK_END);
  mm_file->file_size = ftell(mm_file->file);

  void* result = mmap(nullptr, mm_file->file_size, PROT_READ, MAP_SHARED, fileno(mm_file->file), 0);
  if (result == MAP_FAILED) {
    return nullptr;
  }

  mm_file->mm = reinterpret_cast<uint8_t*>(result);

  return mm_file;
}

struct ScalarContext {

  ScalarContext(std::string data_dir, float_t selectivity) {
    FILE* meta = fopen((data_dir + meta_name).c_str(), "r");
    fread(&input_num, sizeof(input_num), 1, meta);
    fread(&header_pos, sizeof(header_pos), 1, meta);
    fclose(meta);

    std::cout << "input num: " << input_num << std::endl;
    std::cout << "header pos: " << header_pos << std::endl;

    lower_bound = input_num / 2 - (input_num * selectivity / 2);
    upper_bound = input_num / 2 + (input_num * selectivity / 2);

//    Int32Builder builder1(default_memory_pool(), INT32);
//    FloatBuilder builder2(default_memory_pool(), FLOAT);
//
//    std::shared_ptr<MemoryMappedFile> src;
    Status s = MemoryMappedFile::Open(data_dir + data_name, FileMode::READ, &src);
    assert(s.ok());
//    std::shared_ptr<RecordBatchReader> reader;
//    s = RecordBatchReader::Open(src.get(), header_pos, &reader);
//    assert(s.ok());
//    std::shared_ptr<RecordBatch> row_batch;
//    reader->GetRecordBatch(schema, &row_batch);
//
//    builder1.Append(reinterpret_cast<Int32Array *>(row_batch->column(0).get())->raw_data(), input_num);
//    builder2.Append(reinterpret_cast<FloatArray *>(row_batch->column(1).get())->raw_data(), input_num);
//
//    src->Close();
//
//    key_in = builder1.Finish();
//    payload_in = builder2.Finish();

    key_out = new int32_t[input_num];
    payload_out = new float_t[input_num];
  }

  ~ScalarContext() {
    src->Close();
    delete [] payload_out;
    delete [] key_out;
  }

  int32_t input_num;

  float_t lower_bound;
  float_t upper_bound;

  std::shared_ptr<MemoryMappedFile> src;
  int64_t header_pos;
//  std::shared_ptr<RowBatch> row_batch;
//  std::shared_ptr<Array> key_in;
//  std::shared_ptr<Array> payload_in;

  int32_t* key_out;
  float_t* payload_out;
};

auto generate_file(std::string data_dir, int32_t input_num) -> void;
auto scalar_branch(std::shared_ptr<ScalarContext> context) -> size_t;
auto scalar_branchless(std::shared_ptr<ScalarContext> context) -> size_t;
auto run_vector(std::shared_ptr<ScalarContext> context) -> size_t;
void CreateFile(const std::string path, int32_t size);
void MakeRandomInt32Array(std::mt19937 gen, int32_t length, MemoryPool* pool, std::shared_ptr<Array> &array);
void MakeRandomFloatArray(std::mt19937 gen, int32_t length, MemoryPool* pool, std::shared_ptr<Array> &array);


auto main(const int argc, char *argv[]) -> int32_t {
  int32_t input_num;
  double_t selectivity;
  std::string directory;

  po::options_description desc("arrow_test\nusage");

  desc.add_options()
      ("help,h", "Display this help message.")

      ("path,p",
       po::value<std::string>(&directory)->value_name("DATA_PATH")->default_value("./"),
       "Directory path to data file.")

      ("generate,g",
       po::value<int32_t>(&input_num)->value_name("INPUT_NUM"),
       "Generate data file.")

      ("scalar_branch,b",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run scalar branch with a given selectivity")

      ("scalar_branchless,l",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run scalar branchless with a given selectivity")

      ("vector,v",
       po::value<double_t>(&selectivity)->value_name("SELECTIVITY"),
       "Run vector with a given selectivity");

  po::variables_map vm;
  auto optional_style = po::command_line_style::unix_style;

  po::store(po::parse_command_line(argc, argv, desc, optional_style), vm);
  po::notify(vm);

  if (vm.count("generate")) {
    generate_file(directory, input_num);

  } else if (vm.count("scalar_branch")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(directory, selectivity);

    auto start = std::chrono::steady_clock::now();

    size_t out_num = scalar_branch(context);

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "Done! " << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

    std::cout << "# of outputs: " << (out_num) << std::endl;
    std::cout << "selectivity: " << (out_num * 100 / (double_t)context->input_num) << "%\n";

  } else if (vm.count("scalar_branchless")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(directory, selectivity);

    auto start = std::chrono::steady_clock::now();

    size_t out_num = scalar_branchless(context);

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "Done! " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

    std::cout << "# of outputs: " << (out_num) << std::endl;
    std::cout << "selectivity: " << (out_num * 100 / (double_t) context->input_num) << "%\n";

  } else if (vm.count("vector")) {
    std::shared_ptr<ScalarContext> context = std::make_shared<ScalarContext>(directory, selectivity);

    auto start = std::chrono::steady_clock::now();

    size_t out_num = run_vector(context);

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "Done! " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

    std::cout << "# of outputs: " << (out_num) << std::endl;
    std::cout << "selectivity: " << (out_num * 100 / (double_t) context->input_num) << "%\n";
  } else {
    std::cout << desc << "\n";

  }

  return 0;

}

auto scalar_branch(std::shared_ptr<ScalarContext> context) -> size_t {
  int32_t input_num = context->input_num;
  double_t lower_bound = context->lower_bound;
  double_t upper_bound = context->upper_bound;
  int32_t* key_out = context->key_out;
  float_t* payload_out = context->payload_out;

  size_t result_num = 0;

  std::shared_ptr<RecordBatchReader> reader;
  Status s = RecordBatchReader::Open(context->src.get(), context->header_pos, &reader);
  assert(s.ok());
  std::shared_ptr<RecordBatch> row_batch;

  int i = 0;
  while (i < input_num) {
    s = reader->GetRecordBatch(schema, &row_batch);
    if (!s.ok()) {
      std::cerr << s.ToString() << std::endl;
      assert(s.ok());
    }

//    Int32Array* key_arr = reinterpret_cast<Int32Array*>(context->key_in.get());
//    FloatArray* payload_arr = reinterpret_cast<FloatArray*>(context->payload_in.get());

    Int32Array * key_arr = reinterpret_cast<Int32Array *>(row_batch->column(0).get());
    FloatArray * payload_arr = reinterpret_cast<FloatArray *>(row_batch->column(1).get());

    int32_t loop_len = row_batch->num_rows();
    for (int j = 0; j < loop_len; j++) {
      int32_t key = key_arr->Value(j);

      if (key >= lower_bound && key < upper_bound) {
        key_out[result_num] = key;
        payload_out[result_num] = payload_arr->Value(j);
        result_num++;
      }
    }

//    i += batch->num_rows();
    i += loop_len;
  }

  return result_num;
}

auto scalar_branchless(std::shared_ptr<ScalarContext> context) -> size_t {
  int32_t input_num = context->input_num;
  double_t lower_bound = context->lower_bound;
  double_t upper_bound = context->upper_bound;
  int32_t* key_out = context->key_out;
  float_t* payload_out = context->payload_out;

  size_t result_num = 0;

  std::shared_ptr<RecordBatchReader> reader;
  Status s = RecordBatchReader::Open(context->src.get(), context->header_pos, &reader);
  assert(s.ok());
  std::shared_ptr<RecordBatch> row_batch;

  int i = 0;
  while (i < input_num) {
    s = reader->GetRecordBatch(schema, &row_batch);
    if (!s.ok()) {
      std::cerr << s.ToString() << std::endl;
      assert(s.ok());
    }

    Int32Array* key_arr = reinterpret_cast<Int32Array*>(row_batch->column(0).get());
    FloatArray* payload_arr = reinterpret_cast<FloatArray*>(row_batch->column(1).get());
//    Int32Array* key_arr = reinterpret_cast<Int32Array*>(context->key_in.get());
//    FloatArray* payload_arr = reinterpret_cast<FloatArray*>(context->payload_in.get());

    int32_t loop_len = row_batch->num_rows();
    for (int j = 0; j < loop_len; j++) {
      int32_t key = key_arr->Value(j);

      key_out[result_num] = key;
      payload_out[result_num] = payload_arr->Value(j);
      int m = (key >= lower_bound ? 1 : 0) & (key < upper_bound ? 1 : 0);
      result_num += m;
    }

    i += loop_len;

  }

  return result_num;
}

static const size_t RID_BUF_SIZE = 8 * 1024;

auto prepare_perm_mat(__m128i* perm) -> __m128i* {
  for (uint16_t i = 0; i < 256; i++) {
    std::bitset<8> bits(i);
    std::vector<uint16_t> on;
    std::vector<uint16_t> off;

    for (uint16_t j = 0; j < 8; j++) {
      if (bits[j]) {
        on.push_back(j);
      } else {
        off.push_back(j);
      }
    }

    uint16_t a[8];
    int j = 0;
    for (std::vector<uint16_t>::iterator it = on.begin(); it != on.end(); ++it, ++j) {
      a[j] = *it;
    }
    for (std::vector<uint16_t>::iterator it = off.begin(); it != off.end(); ++it, ++j) {
      a[j] = *it;
    }
    perm[i] = _mm_set_epi16(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
  }
  return perm;
}

auto perform_vector(__m256i rid, __m128i *perm_mat, __m256i key, __m256 lb, __m256 ub,
                    int32_t* rid_buf, size_t& buf_idx, size_t& buf_start_idx,
                    const int32_t* key_in, const float_t* payload_in, int32_t* key_out, float_t* payload_out) -> void {

  __m256 cvt_key = _mm256_cvtepi32_ps(key);

  // Unordered compare checks that either inputs are NaN or not. Ordered compare checks that neither inputs are NaN.
  __m256 lb_cmp = _mm256_cmp_ps(lb, cvt_key, _CMP_NGT_US); // If lb is less than key, result is 0xFFFFFFFF. Otherwise, 0.
  __m256 ub_cmp = _mm256_cmp_ps(cvt_key, ub, _CMP_NGE_US);
  __m256 cmp = _mm256_and_ps(lb_cmp, ub_cmp); // if any element is 0xFFFFFFFF, then key satisfies the predicates.

  __mmask8 mask = (__mmask8) _mm256_movemask_ps(cmp);

  if (mask > 0 /* if any bit is set */) {
    // selective store
    __m128i perm_comp = perm_mat[mask];
    __m256i perm = _mm256_cvtepi16_epi32(perm_comp);

    // permute and store the input pointers
    __m256i cvt_cmp = _mm256_cvtps_epi32(cmp);
    cvt_cmp = _mm256_permutevar8x32_epi32(cvt_cmp, perm);
    __m256i ptr = _mm256_permutevar8x32_epi32(rid, perm);

    _mm256_maskstore_epi32(&rid_buf[buf_idx], cvt_cmp, ptr);

    buf_idx += _mm_popcnt_u64(mask);

    // if the buffer is full, flush the buffer
    if (buf_idx + 8 > RID_BUF_SIZE) {
      size_t b;
      for (b = 0; b + 8 < buf_idx; b += 8) {
        // dereference column values and store
        __m256i load_ptr = _mm256_load_si256(reinterpret_cast<__m256i*>(&rid_buf[b]));
        __m256i gather_key = _mm256_i32gather_epi32(key_in, load_ptr, 4);
        __m256 gather_pay = _mm256_i32gather_ps(payload_in, load_ptr, 4);

        // streaming store
        _mm256_stream_si256(reinterpret_cast<__m256i*>(&key_out[b + buf_start_idx]), gather_key);
        _mm256_stream_ps(&payload_out[b + buf_start_idx], gather_pay);
      }

      // Move extra items to the start of the buffer
      ptr = _mm256_load_si256(reinterpret_cast<__m256i*>(&rid_buf[b]));
      _mm256_store_si256(reinterpret_cast<__m256i*>(&rid_buf[0]), ptr);

      buf_start_idx += b;
      buf_idx -= b;
    }
  }

}

auto run_vector(std::shared_ptr<ScalarContext> context) -> size_t {
  // extracting one bit at a time from the bitmask, or use vector selective stores
  // early vs late materialization

  int32_t input_num = context->input_num;
  float_t lower_bound = context->lower_bound;
  float_t upper_bound = context->upper_bound;

  size_t key_out_sz = sizeof(int32_t) * input_num;
  void* key_out_buf = context->key_out;
  int32_t* key_out = reinterpret_cast<int32_t *>(std::align(32, // for streaming store
                                                            4,  // size of int32_t
                                                            key_out_buf,
                                                            key_out_sz));

  size_t pay_out_sz = sizeof(float_t) * input_num;
  void* pay_out_buf = context->payload_out;
  float_t* payload_out = reinterpret_cast<float_t*>(std::align(32, // for streaming store
                                                               4,  // size of float_t
                                                               pay_out_buf,
                                                               pay_out_sz));

  __m256 lb = _mm256_broadcast_ss(&lower_bound);
  __m256 ub = _mm256_broadcast_ss(&upper_bound);

  int32_t rid_buf[RID_BUF_SIZE]; // TODO: should be cache resident
  size_t buf_idx = 0;
  size_t buf_start_idx = 0;

  __m256i v_rid = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  const __m256i v_eight = _mm256_set_epi32(8, 8, 8, 8, 8, 8, 8, 8);

  std::shared_ptr<RecordBatchReader> reader;
  Status s = RecordBatchReader::Open(context->src.get(), context->header_pos, &reader);
  assert(s.ok());
  std::shared_ptr<RecordBatch> row_batch;

  int i = 0;
  while (i < input_num) {
    s = reader->GetRecordBatch(schema, &row_batch);
    if (!s.ok()) {
      std::cerr << s.ToString() << std::endl;
      assert(s.ok());
    }

    Int32Array *key_arr = reinterpret_cast<Int32Array *>(row_batch->column(0).get());
    FloatArray *payload_arr = reinterpret_cast<FloatArray *>(row_batch->column(1).get());
//    Int32Array* key_arr = reinterpret_cast<Int32Array*>(context->key_in.get());
//    FloatArray* payload_arr = reinterpret_cast<FloatArray*>(context->payload_in.get());

    const int32_t *key_in = key_arr->raw_data();
    const float_t *payload_in = payload_arr->raw_data();

    int32_t array_len = key_arr->length();
    const int32_t remain_iter = array_len % 8;
    const int32_t max_iter = array_len - remain_iter;

    __m128i perm_mat[256];
    prepare_perm_mat(perm_mat);

    size_t rid;
    for (rid = 0; rid < max_iter; rid += 8) {
      __m256i v_key = _mm256_load_si256(reinterpret_cast<const __m256i *>(&key_in[rid]));
      perform_vector(v_rid, perm_mat, v_key, lb, ub, rid_buf, buf_idx, buf_start_idx, key_in, payload_in, key_out,
                     payload_out);

      v_rid = _mm256_add_epi32(v_rid, v_eight);
    }

    // evaluate remaining keys
    int32_t remain_key_arr[8] = {-1};
    for (int i = 0; i < remain_iter; i++) {
      remain_key_arr[i] = key_in[rid + i];
    }
    __m256i remain_key = _mm256_set_epi32(remain_key_arr[7], remain_key_arr[6], remain_key_arr[5], remain_key_arr[4],
                                          remain_key_arr[3], remain_key_arr[2], remain_key_arr[1], remain_key_arr[0]);
    // TODO: mask?
    perform_vector(v_rid, perm_mat, remain_key, lb, ub, rid_buf, buf_idx, buf_start_idx, key_in, payload_in, key_out,
                   payload_out);

    // flush remaining items in the buffer
    size_t b = 0;
    for (b = 0; b + 8 < buf_idx; b += 8) {
      // dereference column values and store
      __m256i ptr = _mm256_load_si256(reinterpret_cast<__m256i *>(&rid_buf[b]));
      __m256i key = _mm256_i32gather_epi32(key_in, ptr, 4);
      __m256 pay = _mm256_i32gather_ps(payload_in, ptr, 4);

      // streaming store
      _mm256_stream_si256(reinterpret_cast<__m256i *>(&key_out[b + buf_start_idx]), key);
      _mm256_stream_ps(&payload_out[b + buf_start_idx], pay);
    }

    // flush remaining items in the buffer
    uint32_t v_mask[8] = {0};
    for (int i = 0; i < buf_idx - b; i++) {
      v_mask[i] = 0xFFFFFFFF;
    }
    __m256i remain_mask = _mm256_set_epi32(v_mask[7], v_mask[6], v_mask[5], v_mask[4], v_mask[3], v_mask[2], v_mask[1],
                                           v_mask[0]);
    const __m256i zero = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);

    __m256i ptr = _mm256_maskload_epi32(&rid_buf[b], remain_mask);
    __m256i key = _mm256_mask_i32gather_epi32(zero, key_in, ptr, remain_mask, 4);
    __m256 pay = _mm256_mask_i32gather_ps(zero, payload_in, ptr, remain_mask, 4);

    _mm256_maskstore_epi32(&key_out[b + buf_start_idx], remain_mask, key);
    _mm256_maskstore_ps(&payload_out[b + buf_start_idx], remain_mask, pay);

//    i += batch->num_rows();
    i += key_arr->length();
  }

  return buf_start_idx + buf_idx;
}

auto generate_file(std::string data_dir, int32_t input_num) -> void {

  // Random generation
  std::random_device rd;
  std::mt19937 gen(rd());

  std::shared_ptr<Array> a0, a1;
  MemoryPool* pool = default_memory_pool();

  MakeRandomInt32Array(gen, input_num, pool, a0);
  MakeRandomFloatArray(gen, input_num, pool, a1);

  RecordBatch* row_batch = new RecordBatch(schema, input_num, {a0, a1});
  int64_t row_batch_size;
  Status s = GetRecordBatchSize(row_batch, &row_batch_size);
  assert(s.ok());
  std::cout << "row batch size: " << row_batch_size << std::endl;

  std::string path = data_dir + data_name;
  CreateFile(path, row_batch_size);
  std::cout << "input: " << input_num << std::endl;

  std::shared_ptr<MemoryMappedFile> mmap;
  s = MemoryMappedFile::Open(path, FileMode::READWRITE, &mmap);
  assert(s.ok());

  int64_t body_end_offset;
  int64_t header_pos;
  s = WriteRecordBatch(row_batch->columns(), row_batch->num_rows(),
      mmap.get(), &body_end_offset, &header_pos);
  assert(s.ok());
  mmap->Close();
  delete row_batch;

  FILE* meta = fopen((data_dir + meta_name).c_str(), "w");
  fwrite(&input_num, sizeof(input_num), 1, meta);
  fwrite(&header_pos, sizeof(int64_t), 1, meta);
  fclose(meta);

  if (!s.ok()) {
    std::cerr << s.ToString() << std::endl;
    exit(1);
  } else {
    int64_t size;
    mmap->GetSize(&size);
    std::cout << size << " bytes written\n";
    std::cout << "header_pos: " << header_pos << std::endl;
  }
}

void MakeRandomInt32Array(std::mt19937 gen, int32_t length, MemoryPool* pool, std::shared_ptr<Array> &array) {
  std::uniform_int_distribution<> idis(0, length - 1);
  std::shared_ptr<PoolBuffer> data = std::make_shared<PoolBuffer>(pool);
  data->Resize(length * sizeof(int32_t));
  auto pData = reinterpret_cast<int32_t *>(data->mutable_data());
  for (int i = 0; i < length; i++) {
    pData[i] = idis(gen);
  }

  Int32Builder builder(pool, INT32);
  builder.Append(reinterpret_cast<const int32_t *>(data->data()), length);

  array = builder.Finish();
}

void MakeRandomFloatArray(std::mt19937 gen, int32_t length, MemoryPool* pool, std::shared_ptr<Array> &array) {
  std::uniform_real_distribution<float_t> fdis(0, 1);
  std::shared_ptr<PoolBuffer> data = std::make_shared<PoolBuffer>(pool);
  data->Resize(length * sizeof(float_t));
  auto pData = reinterpret_cast<float_t *>(data->mutable_data());
  for (int i = 0; i < length; i++) {
    pData[i] = fdis(gen);
  }

  FloatBuilder builder(pool, FLOAT);
  builder.Append(reinterpret_cast<const float_t *>(data->data()), length);

  array = builder.Finish();
}

void CreateFile(const std::string path, int32_t size) {
  FILE* file = fopen(path.c_str(), "w");
  ftruncate(fileno(file), size);
  fclose(file);
}