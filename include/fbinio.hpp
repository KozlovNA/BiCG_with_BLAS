#ifndef FBINIO_HPP
#define FBINIO_HPP

# include <fstream>
# include <Eigen/Dense>

//binary output for Eigen::MatrixX<DataType>
template<class Matrix>
void read_binary(const char* filename,
                 Matrix& matrix)
{
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  uint32_t rows=0, cols=0;
  in.read((char*) (&rows),sizeof(uint32_t));
  in.read((char*) (&cols),sizeof(uint32_t));
  matrix.resize(rows, cols);
  in.read((char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar));
  in.close();
}

// only for vectors now
template<class VT>
void write_binary(  const char         *filename, 
                    const VT           &output,
                    const u_int32_t    &rows,
                    const u_int32_t    &cols       )
{
  using T = std::decay<decltype(*output.begin())>::type;
  std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
  out.write((char*) (&rows), sizeof(uint32_t));
  out.write((char*) (&cols), sizeof(uint32_t));
  out.write((char*) output.data(), rows*cols*sizeof(T));
  out.close();
}


#endif