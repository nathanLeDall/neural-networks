use activations::SIGMOID;
use network::Network;
use std::io;
pub mod activations;
pub mod matrix;
pub mod network;
fn main() {
    let inputs = vec![
		vec![0.0, 0.0],
		vec![0.0, 1.0],
		vec![1.0, 0.0],
		vec![1.0, 1.0],
	];
	let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

	let mut network = Network::new(vec![2, 3, 1], 0.5, SIGMOID);
	
	network.train(inputs, targets, 100000);

	println!("{:?}", network.feed_forward(vec![0.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![0.0, 1.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 1.0]));
	let mut user_input = String::new();
	let mut user_input2 = String::new();
	print!("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	io::stdin().read_line(&mut user_input).expect("Failed to read line");
	io::stdin().read_line(&mut user_input2).expect("Failed to read line");
	let user_number: f64 = user_input.trim().parse().expect("Invalid input");
	let user_number2: f64 = user_input2.trim().parse().expect("Invalid input");

	println!("{:?}", network.feed_forward(vec![user_number, user_number2]))
}
