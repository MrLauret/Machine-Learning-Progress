use rand::random_range;

const LEARNING_RATE: f32 = 0.1;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(output: f32) -> f32 {
    output * (1.0 - output)
}

struct XorNetwork {
    or_gate: Neuron,
    nand_gate: Neuron,
    and_gate: Neuron,
}

struct Neuron {
    pub weight_a: f32,
    pub weight_b: f32,
    pub bias: f32
}

impl Neuron {
    pub fn predict(&self, a: f32, b: f32) -> f32 {
        let weighted_sum = (a * self.weight_a) + (b * self.weight_b);
        let raw_score = weighted_sum + self.bias;
        let prediction = sigmoid(raw_score);

        prediction
    }

    pub fn init() -> Neuron {
        Neuron {
            weight_a: random_range(-1.0..=1.0),
            weight_b: random_range(-1.0..=1.0),
            bias: random_range(-1.0..=1.0),
        }
    }
}

impl XorNetwork {
    pub fn init() -> XorNetwork {
        XorNetwork {
            or_gate: Neuron::init(),
            nand_gate: Neuron::init(),
            and_gate: Neuron::init()
        }
    }

    pub fn predict(&self, a: f32, b: f32) -> f32 {
        let or_out = self.or_gate.predict(a, b);
        let nand_out = self.nand_gate.predict(a, b);

        self.and_gate.predict(or_out, nand_out)
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f32>>, epochs: usize) {
        for _ in 0..epochs {
            for row in inputs {
                let a = row[0];
                let b = row[1];
                let target = row[2];

                let or_out = self.or_gate.predict(a, b);
                let nand_out = self.nand_gate.predict(a, b);
                let prediction = self.and_gate.predict(or_out, nand_out);

                let output_delta = (target - prediction) * sigmoid_derivative(prediction);

                let or_error = output_delta * self.and_gate.weight_a;
                let nand_error = output_delta * self.and_gate.weight_b;

                let or_delta = or_error * sigmoid_derivative(or_out);
                let nand_delta = nand_error * sigmoid_derivative(nand_out);

                self.and_gate.weight_a += or_out * output_delta * LEARNING_RATE;
                self.and_gate.weight_b += nand_out * output_delta * LEARNING_RATE;
                self.and_gate.bias += output_delta * LEARNING_RATE;
                
                self.or_gate.weight_a += a * or_delta * LEARNING_RATE;
                self.or_gate.weight_b += b * or_delta * LEARNING_RATE;
                self.or_gate.bias += or_delta * LEARNING_RATE;

                self.nand_gate.weight_a += a * nand_delta * LEARNING_RATE;
                self.nand_gate.weight_b += b * nand_delta * LEARNING_RATE;
                self.nand_gate.bias += nand_delta * LEARNING_RATE;
            }
        }

        println!("\n");
        println!("TRAINING COMPLETED");
    }
}

fn main() {
    let xor_input: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let mut xor_network = XorNetwork::init();

    println!("\n");
    for row in &xor_input {
        println!("{:.2}%", xor_network.predict(row[0], row[1]) * 100.0)
    }

    xor_network.train(&xor_input, 10000000);
    
    println!("\n");
    for row in &xor_input {
        println!("{:.2}%", xor_network.predict(row[0], row[1]) * 100.0)
    }
}
