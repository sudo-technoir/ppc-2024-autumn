#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>
#include <vector>

#include "seq/oturin_a_image_smoothing/include/ops_seq.hpp"

TEST(oturin_a_image_smoothing_seq_functest, Test_IMAGE_LINE) {
  std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
  std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
  std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

  std::string filenameOriginal = dir_path + "/../line.bmp";
  std::string filenameCompare = dir_path + "/../lineREF.bmp";

  int width{};
  int height{};

  // Create data
  std::vector<uint8_t> in = oturin_a_image_smoothing_seq::ReadBMP(filenameOriginal.c_str(), width, height);
  std::vector<uint8_t> ref = oturin_a_image_smoothing_seq::ReadBMP(filenameCompare.c_str(), width, height);
  std::vector<uint8_t> out(width * height * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_image_smoothing_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true) << filenameOriginal;
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}

TEST(oturin_a_image_smoothing_seq_functest, Test_IMAGE_CIRCLE) {
  std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
  std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
  std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

  std::string filenameOriginal = dir_path + "/../circle.bmp";
  std::string filenameCompare = dir_path + "/../circleREF.bmp";

  if (!std::filesystem::exists(filenameOriginal)) {
    ASSERT_EQ(true, false) << "file not found" << std::endl;
  }

  int width{};
  int height{};

  // Create data
  std::vector<uint8_t> in = oturin_a_image_smoothing_seq::ReadBMP(filenameOriginal.c_str(), width, height);
  std::vector<uint8_t> ref = oturin_a_image_smoothing_seq::ReadBMP(filenameCompare.c_str(), width, height);
  std::vector<uint8_t> out(width * height * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_image_smoothing_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true) << filenameOriginal << ' ' << filenameCompare;
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

#if __APPLE__
  ASSERT_EQ(ref.size(), out.size());
  for (unsigned long i = 0; i < ref.size(); i++) {
    EXPECT_NEAR(ref[i], out[i], 1e-0) << i << ' ';
  }
#else
  ASSERT_EQ(ref, out) << width << ' ' << height << ' ' << filenameOriginal << ' ' << filenameCompare;
#endif
}

TEST(oturin_a_image_smoothing_seq_functest, Test_IMAGE_COLOR) {
  std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
  std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
  std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

  std::string filenameOriginal = dir_path + "/../color.bmp";
  std::string filenameCompare = dir_path + "/../colorREF.bmp";

  int width{};
  int height{};

  // Create data
  std::vector<uint8_t> in = oturin_a_image_smoothing_seq::ReadBMP(filenameOriginal.c_str(), width, height);
  std::vector<uint8_t> ref = oturin_a_image_smoothing_seq::ReadBMP(filenameCompare.c_str(), width, height);
  std::vector<uint8_t> out(width * height * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_image_smoothing_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true) << filenameOriginal;
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

#if __APPLE__
  ASSERT_EQ(ref.size(), out.size());
  for (unsigned long i = 0; i < ref.size(); i++) {
    EXPECT_NEAR(ref[i], out[i], 1e-0) << i << ' ';
  }
#else
  ASSERT_EQ(ref, out) << width << ' ' << height << ' ' << filenameOriginal << ' ' << filenameCompare;
#endif
}