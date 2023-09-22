
use rand::{thread_rng, Rng};
#[derive(Clone)]
pub struct Matrix {
	pub columns: usize,
	pub rows: usize,
	pub data: Vec<Vec<f64>>,
}

impl Matrix{
	pub fn from(data: Vec<Vec<f64>>) -> Matrix {
		Matrix {
			rows: data.len(),
			columns: data[0].len(),
			data,
		}
	}
	pub fn zeros(c: usize, r: usize) -> Matrix {
		Matrix{
			columns: c,
			rows: r,
			data: vec![vec![0.0; c]; r],
		}
	}
	pub fn rand_matrix(c: usize, r: usize) -> Matrix {
		let mut rng = thread_rng();

		let mut res = Matrix::zeros(c, r);
		for i in 0..r {
			for j in 0..c {
				res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
			}
		}
		res
	}
	pub fn dot_mul(&self, other: &Matrix) -> Matrix{
		if self.columns != other.rows {
			panic!("rows not equal columns");
		}
		let mut res = Matrix::zeros(other.columns, self.rows);
		for i in 0..self.rows {
			for k in 0..other.columns {
				let mut num = 0.0;
				for j in 0..self.columns {
					num += self.data[i][j] * other.data[j][k];
				}
				res.data[i][k] = num;
			}
		}
		res
	}
	pub fn add_matrix(&self, other: &Matrix) -> Matrix{
		if other.rows != self.rows || other.columns != self.columns{
			panic!("matrices are not of the same size");
		}
		let mut res = Matrix::zeros(self.columns, self.rows);
		for i in 0..self.rows{
			for j in 0..self.columns{
				res.data[i][j] = self.data[i][j] + other.data[i][j];
			}
		}
		res
	}

	pub fn multiply_matrix(&self, other: &Matrix) -> Matrix{
		if self.columns != other.columns || self.rows != other.rows{
			panic!("wrong sized matrix");
		}
		let mut res = Matrix::zeros(self.columns, self.rows);
		for i in 0..self.rows{
			for j in 0..self.columns{
				res.data[i][j] = self.data[i][j] * other.data[i][j];
			}
		}
		res
	}
	pub fn rotate_matrix(&self) -> Matrix{
		let mut res = Matrix::zeros(self.rows, self.columns);
		for i in 0..self.rows{
			for j in 0..self.columns{
				res.data[j][i] = self.data[i][j];
			}
		}
		res
	}
	pub fn map(&self, function: &dyn Fn(f64) -> f64) -> Matrix{
		Matrix::from(
			(self.data)
			.clone()
			.into_iter()
			.map(|row| row.into_iter().map(|value| function(value)).collect())
			.collect(),
		)
	}
	pub fn subtract_matrix(&self, other: &Matrix) -> Matrix{
		if self.columns != other.columns || self.rows != other.rows{
			panic!("wrong sized matrix");
		}
		let mut res = Matrix::zeros(self.columns, self.rows);
		for i in 0..self.rows{
			for j in 0..self.columns{
				res.data[i][j] = self.data[i][j] - other.data[i][j];
			}
		}
		res
	}
}