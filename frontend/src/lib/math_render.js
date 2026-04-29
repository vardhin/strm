// Render a symbolic expression from a model's function metadata.
// model info from GET /models/{name}:
//   functions: [{id, name, arity, layer, composition?, constants?, const_mode?}]
//
// Composition entries: {func_id, func_name, args} where args index into
// the "available_values" list that starts with inputs x0..x(arity-1) and
// grows with each step's result appended.

const INFIX_BINARY = {
	ADD: '+',
	SUB: '−',
	MUL: '·',
	DIV: '/',
	OR: '∨',
	AND: '∧',
	LT: '<',
	LTE: '≤',
	GT: '>',
	GTE: '≥',
	EQ: '=',
	NEQ: '≠'
};

const PREC = {
	atom: 100,
	'^': 70,
	'·': 60,
	'/': 60,
	'+': 40,
	'−': 40,
	'<': 30,
	'≤': 30,
	'>': 30,
	'≥': 30,
	'=': 25,
	'≠': 25,
	'∧': 20,
	'∨': 15,
	fn: 90
};

function wrap(expr, myPrec, parentPrec) {
	if (myPrec < parentPrec) return `(${expr})`;
	return expr;
}

// Build a lookup from function name -> function meta (from /models/{name})
export function functionsByName(fns) {
	const map = {};
	for (const f of fns) map[f.name] = f;
	return map;
}

export function functionsById(fns) {
	const map = {};
	for (const f of fns) map[f.id] = f;
	return map;
}

// Render a function as a symbolic expression in variables x0..x(arity-1)
// Uses parentPrec for grouping.
export function renderFunction(fn, allFns, opts = {}) {
	const { format = 'ascii', parentPrec = 0, maxDepth = 12 } = opts;
	if (maxDepth <= 0) return '…';

	const vars = [];
	for (let i = 0; i < fn.arity; i++) vars.push(varName(i, format));

	if (fn.layer === 0) {
		// Primitive — show as f(x0, x1, ...)
		return renderPrimitiveCall(fn.name, vars, format, parentPrec);
	}

	if (!fn.composition) {
		return `${fn.name}(${vars.join(', ')})`;
	}

	// Symbolic evaluation: walk composition, build expression strings in
	// the "available" array
	const byName = functionsByName(allFns);
	const available = vars.map((v) => ({ str: v, prec: PREC.atom }));

	for (const step of fn.composition) {
		const childMeta = byName[step.func_name];
		const args = step.args;

		let res;
		if (childMeta && childMeta.layer === 0) {
			res = applyPrimitive(step.func_name, args, available, format);
		} else if (childMeta) {
			// Learned child — inline render by substituting
			res = applyLearned(childMeta, args, available, allFns, format, maxDepth - 1);
		} else {
			const argStrs = args.map((i) => (i === -1 ? '0' : available[i].str));
			res = { str: `${step.func_name}(${argStrs.join(', ')})`, prec: PREC.fn };
		}
		available.push(res);
	}

	let result = available[available.length - 1];

	if (fn.constants && fn.constants.length > 0) {
		const c = fn.constants[0];
		if (fn.const_mode === 'additive') {
			result = {
				str: `${result.str} + ${formatConst(c, format)}`,
				prec: PREC['+']
			};
		} else {
			result = {
				str: `${formatConst(c, format)} · ${wrap(result.str, result.prec, PREC['·'])}`,
				prec: PREC['·']
			};
		}
	}

	return wrap(result.str, result.prec, parentPrec);
}

function varName(i, format) {
	if (format === 'latex') {
		return `x_{${i}}`;
	}
	return `x${subscript(i)}`;
}

function subscript(i) {
	const SUBS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
	return String(i)
		.split('')
		.map((d) => SUBS[+d])
		.join('');
}

function formatConst(c, format) {
	if (Number.isInteger(c)) return String(c);
	return c.toFixed(4).replace(/\.?0+$/, '');
}

// Apply a primitive step to "available" and return new expression
function applyPrimitive(name, args, available, format) {
	const values = args.map((i) => (i === -1 ? { str: '0', prec: PREC.atom } : available[i]));

	// Binary infix operators
	if (INFIX_BINARY[name] && values.length === 2) {
		const op = INFIX_BINARY[name];
		const p = PREC[op];
		const lhs = wrap(values[0].str, values[0].prec, p);
		const rhs = wrap(values[1].str, values[1].prec, p + 1); // right-wrap a bit tighter
		return { str: `${lhs} ${op} ${rhs}`, prec: p };
	}

	// Unary special cases
	if (name === 'SQUARE') {
		const a = wrap(values[0].str, values[0].prec, PREC['^']);
		return { str: `${a}²`, prec: PREC['^'] };
	}
	if (name === 'INC') {
		return { str: `${wrap(values[0].str, values[0].prec, PREC['+'])} + 1`, prec: PREC['+'] };
	}
	if (name === 'DEC') {
		return { str: `${wrap(values[0].str, values[0].prec, PREC['+'])} − 1`, prec: PREC['+'] };
	}
	if (name === 'NOT') {
		return { str: `¬${wrap(values[0].str, values[0].prec, PREC.fn)}`, prec: PREC.fn };
	}
	if (name === 'ABS') {
		return { str: `|${values[0].str}|`, prec: PREC.atom };
	}
	if (name === 'MULTIPLICATIVE_INV') {
		return {
			str: `1/${wrap(values[0].str, values[0].prec, PREC['/'] + 1)}`,
			prec: PREC['/']
		};
	}
	if (name === 'CONST') {
		return values[0];
	}
	if (name === 'NULL') {
		return { str: '0', prec: PREC.atom };
	}
	if (name === 'COND' && values.length === 3) {
		return {
			str: `(${values[0].str} ? ${values[1].str} : ${values[2].str})`,
			prec: PREC.atom
		};
	}
	if (name === 'LOOP') {
		// args: [body_fn_id, count_idx, init_idx, (step_idx)]
		const body = `#${args[0]}`;
		const count = available[args[1]]?.str ?? '?';
		const init = args[2] === -1 ? '0' : available[args[2]]?.str ?? '?';
		const rest = args.length === 4 ? `, ${available[args[3]]?.str}` : '';
		return { str: `LOOP(${body}, ${count}, ${init}${rest})`, prec: PREC.fn };
	}

	// Fallback: functional notation
	return renderPrimitiveCall(name, values.map((v) => v.str), format, PREC.fn);
}

function renderPrimitiveCall(name, argStrs, format, parentPrec) {
	if (argStrs.length === 0) return name;
	return `${name}(${argStrs.join(', ')})`;
}

function applyLearned(childMeta, args, available, allFns, format, maxDepth) {
	// Render the child with variables replaced by the actual arg expressions.
	// Use a substitution pass on a recursive render.
	const substitutedVars = args.map((i) =>
		i === -1 ? { str: '0', prec: PREC.atom } : available[i]
	);

	// We need to render childMeta, but with x_i replaced by substitutedVars[i].str
	// Rebuild an "available" for the child
	const childAvail = substitutedVars.slice();
	const byName = functionsByName(allFns);

	if (!childMeta.composition) {
		// Just inline as a function call
		return {
			str: `${childMeta.name}(${substitutedVars.map((v) => v.str).join(', ')})`,
			prec: PREC.fn
		};
	}

	for (const step of childMeta.composition) {
		const gc = byName[step.func_name];
		let res;
		if (gc && gc.layer === 0) {
			res = applyPrimitive(step.func_name, step.args, childAvail, format);
		} else if (gc) {
			res = applyLearned(gc, step.args, childAvail, allFns, format, maxDepth - 1);
		} else {
			const argStrs = step.args.map((i) => (i === -1 ? '0' : childAvail[i].str));
			res = { str: `${step.func_name}(${argStrs.join(', ')})`, prec: PREC.fn };
		}
		childAvail.push(res);
	}

	let r = childAvail[childAvail.length - 1];
	if (childMeta.constants && childMeta.constants.length > 0) {
		const c = childMeta.constants[0];
		if (childMeta.const_mode === 'additive') {
			r = { str: `${r.str} + ${formatConst(c, format)}`, prec: PREC['+'] };
		} else {
			r = {
				str: `${formatConst(c, format)} · ${wrap(r.str, r.prec, PREC['·'])}`,
				prec: PREC['·']
			};
		}
	}
	return r;
}

// Produce a short human-readable signature like "f(x0, x1) = ..."
export function renderSignature(fn, allFns) {
	const vars = [];
	for (let i = 0; i < fn.arity; i++) vars.push(`x${subscript(i)}`);
	const body = renderFunction(fn, allFns, { format: 'ascii', parentPrec: 0 });
	const head = fn.arity >= 0 ? `${fn.name}(${vars.join(', ')})` : `${fn.name}(...)`;
	return `${head} = ${body}`;
}

// Describe a primitive in natural language
export const PRIMITIVE_DESCRIPTIONS = {
	ADD: 'a + b',
	SUB: 'a − b',
	MUL: 'a · b',
	DIV: 'a / b (0 if b=0)',
	INC: 'a + 1',
	DEC: 'a − 1',
	SQUARE: 'a²',
	MULTIPLICATIVE_INV: '1 / a (0 if a=0)',
	ABS: '|a|',
	OR: 'bitwise a | b',
	AND: 'bitwise a & b',
	NOT: 'bitwise ~a',
	LT: '1 if a < b else 0',
	LTE: '1 if a ≤ b else 0',
	GT: '1 if a > b else 0',
	GTE: '1 if a ≥ b else 0',
	EQ: '1 if a = b else 0',
	NEQ: '1 if a ≠ b else 0',
	COND: 'b if a else c',
	CONST: 'identity (returns a)',
	NULL: '0 (unused column)',
	LOOP: 'iterate body_fn count times',
	WHILE: 'iterate body while cond holds',
	ACCUM: 'count iterations'
};
