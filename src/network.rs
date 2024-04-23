use super::{activations::Activation, matrix::Matrix};
use std::{
	fs::File,
	io::{Read, Write},
};
//use crate::activations::{Activation};
use serde::{Deserialize, Serialize};

pub struct Network<'a> {
	layers: Vec<usize>,
	weights: Vec<Matrix>,
	biases: Vec<Matrix>,
	data: Vec<Matrix>,
	learning_rate: f64,
	activation: Activation<'a>,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
	weights: Vec<Vec<Vec<f64>>>,
	biases: Vec<Vec<Vec<f64>>>,
}

impl Network<'_>{
	pub fn new<'a>(layers: Vec<usize>, learning_rate: f64, activation: Activation<'a>,) -> Network<'a>{
		let mut weights = vec![];
		let mut biases = vec![];
		for i in 0..layers.len()-1{
			let matrix = Matrix::rand_matrix_xavier(layers[i],layers[i+1],layers[i],layers[i+1]);
			println!("{:?}",matrix.data);
			weights.push(matrix);
			biases.push(Matrix::rand_matrix(1, layers[i+1]));
		}
		Network{
			layers: layers,
			weights: weights,
			biases: biases,
			data: vec![],
			learning_rate: learning_rate,
			activation: activation,
		}
	}
	pub fn feed_forward(&mut self, input: Vec<f64>) -> Vec<f64>{
		if input.len() != self.layers[0] {
			panic!("invalid number of inputs");
		}
		let mut current = Matrix::from(vec![input]).rotate_matrix();
		self.data = vec![current.clone()];

		for i in 0..self.layers.len()-1{
			current  = self.weights[i].dot_mul(&current).add_matrix(&self.biases[i]).map(self.activation.function);
			self.data.push(current.clone());
		}
		current.rotate_matrix().data[0].to_owned()
	}
	pub fn back_prob(&mut self, outputs: Vec<f64>, targets: Vec<f64>){
		if targets.len() != self.layers[self.layers.len()-1]{
			panic!("invalid number of target");
		}
		let parsed = Matrix::from(vec![outputs]).rotate_matrix();
		let mut errors = Matrix::from(vec![targets]).rotate_matrix().subtract_matrix(&parsed);
		let mut gradient = parsed.map(self.activation.derivative);

		for i in (0..self.layers.len() - 1).rev() {
			gradient = gradient.multiply_matrix(&errors).map(&|x| x*self.learning_rate);

			self.weights[i] = self.weights[i].add_matrix(&gradient.dot_mul(&self.data[i].rotate_matrix()));
			self.biases[i] = self.biases[i].add_matrix(&gradient);

			errors = self.weights[i].rotate_matrix().dot_mul(&errors);
			gradient = self.data[i].map(self.activation.derivative);
		}
	}
	pub fn train(&mut self, input: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epoch: u32, target_lr: f64) {
		let learn_change = (self.learning_rate - target_lr) / epoch as f64;
		for i in 1..=epoch{
			if epoch<100 || i%(epoch/100) == 0{
				//println! ("epoch {} of {}", i, epoch);
			}
			for j in 0..input.len(){
				let outputs = self.feed_forward(input[j].clone());
				self.back_prob(outputs, targets[j].clone());
			}
			self.learning_rate -= learn_change;
		}
	}
}
