// Independent double-precision direct DFT oracle. Compensated summation and
// reduced integer phases keep reference error small for sampled large DFTs.
export function frequencies(n) {
  if (n <= 128) return Array.from({length: n}, (_, i) => i);
  return [...new Set([0, 1, 2, 3, 7, Math.floor(n / 7), Math.floor(n / 3),
    Math.floor(n / 2), n - 3, n - 2, n - 1])];
}

function sum(values) {
  let total = 0, correction = 0;
  for (const value of values) {
    const next = value - correction, combined = total + next;
    correction = (combined - total) - next; total = combined;
  }
  return total;
}

export function directComplex(input, n, batch, k, inverse = false) {
  const real = [], imag = [];
  for (let j = 0; j < n; ++j) {
    const phase = (inverse ? 2 : -2) * Math.PI * ((j * k) % n) / n;
    const c = Math.cos(phase), s = Math.sin(phase), i = 4 * (batch * n + j);
    const r = input[i] + input[i + 2], v = input[i + 1] + input[i + 3];
    real.push(r * c - v * s); imag.push(r * s + v * c);
  }
  const scale = inverse ? n : 1;
  return [sum(real) / scale, sum(imag) / scale];
}

export function directReal(input, n, batch, k) {
  const real = [], imag = [];
  for (let j = 0; j < n; ++j) {
    const phase = -2 * Math.PI * ((j * k) % n) / n, i = 2 * (batch * n + j);
    const value = input[i] + input[i + 1];
    real.push(value * Math.cos(phase)); imag.push(value * Math.sin(phase));
  }
  return [sum(real), sum(imag)];
}

export function directHermitian(input, n, batch, k) {
  const bins = Math.floor(n / 2) + 1, offset = 4 * batch * bins;
  const values = [input[offset] + input[offset + 2]];
  for (let j = 1; j < bins; ++j) {
    const i = offset + 4 * j, r = input[i] + input[i + 2];
    const v = input[i + 1] + input[i + 3];
    const phase = 2 * Math.PI * ((j * k) % n) / n;
    values.push((2 * j === n ? 1 : 2) * (r * Math.cos(phase) - v * Math.sin(phase)));
  }
  return sum(values) / n;
}

export function samples(n, batchCount, components, paired) {
  const input = new Float32Array(n * batchCount * components);
  let state = 0x89abcdef;
  const random = () => {state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 2**32 - .5;};
  for (let batch = 0; batch < batchCount; ++batch) for (let j = 0; j < n; ++j) {
    const i = components * (batch * n + j);
    if (batch === 0) {
      input[i] = Number(j === Math.floor(n / 3));
    } else if (components === 2) {
      input[i] = random() + .2;
      input[i + 1] = paired ? random() * 1e-8 : 0;
    } else {
      input[i] = random() + .2; input[i + 1] = random();
      input[i + 2] = paired ? random() * 1e-8 : 0;
      input[i + 3] = paired ? random() * 1e-8 : 0;
    }
  }
  return input;
}
