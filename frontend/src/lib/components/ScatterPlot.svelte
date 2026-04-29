<script>
	let {
		points = [],
		xLabel = 'x',
		yLabel = 'y',
		width = 600,
		height = 380,
		pointRadius = 3
	} = $props();

	const margin = { top: 20, right: 20, bottom: 40, left: 55 };
	const innerW = $derived(width - margin.left - margin.right);
	const innerH = $derived(height - margin.top - margin.bottom);

	const bounds = $derived.by(() => {
		if (points.length === 0) return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
		let xMin = Infinity,
			xMax = -Infinity,
			yMin = Infinity,
			yMax = -Infinity;
		for (const [x, y] of points) {
			if (x < xMin) xMin = x;
			if (x > xMax) xMax = x;
			if (y < yMin) yMin = y;
			if (y > yMax) yMax = y;
		}
		if (xMin === xMax) {
			xMin -= 0.5;
			xMax += 0.5;
		}
		if (yMin === yMax) {
			yMin -= 0.5;
			yMax += 0.5;
		}
		return { xMin, xMax, yMin, yMax };
	});

	function sx(x) {
		const { xMin, xMax } = bounds;
		return ((x - xMin) / (xMax - xMin)) * innerW;
	}
	function sy(y) {
		const { yMin, yMax } = bounds;
		return innerH - ((y - yMin) / (yMax - yMin)) * innerH;
	}

	function ticks(min, max, n = 5) {
		const step = (max - min) / (n - 1);
		const arr = [];
		for (let i = 0; i < n; i++) arr.push(min + step * i);
		return arr;
	}

	const xTicks = $derived(ticks(bounds.xMin, bounds.xMax));
	const yTicks = $derived(ticks(bounds.yMin, bounds.yMax));

	function fmt(v) {
		if (Number.isInteger(v)) return String(v);
		const abs = Math.abs(v);
		if (abs >= 1000 || (abs < 0.01 && abs > 0)) return v.toExponential(2);
		return v.toFixed(2);
	}

	let hovered = $state(null);

	function onMouseEnter(i, x, y) {
		hovered = { i, x, y };
	}
	function onMouseLeave() {
		hovered = null;
	}
</script>

<div class="plot-wrap">
	<svg {width} {height} class="plot">
		<g transform="translate({margin.left}, {margin.top})">
			<!-- grid -->
			{#each xTicks as t}
				<line x1={sx(t)} x2={sx(t)} y1={0} y2={innerH} class="grid" />
			{/each}
			{#each yTicks as t}
				<line x1={0} x2={innerW} y1={sy(t)} y2={sy(t)} class="grid" />
			{/each}

			<!-- axes -->
			<line x1={0} x2={innerW} y1={innerH} y2={innerH} class="axis" />
			<line x1={0} x2={0} y1={0} y2={innerH} class="axis" />

			<!-- x ticks -->
			{#each xTicks as t}
				<g transform="translate({sx(t)}, {innerH})">
					<line y1={0} y2={4} class="axis" />
					<text y={18} text-anchor="middle" class="tick">{fmt(t)}</text>
				</g>
			{/each}
			<!-- y ticks -->
			{#each yTicks as t}
				<g transform="translate(0, {sy(t)})">
					<line x1={-4} x2={0} class="axis" />
					<text x={-8} y={4} text-anchor="end" class="tick">{fmt(t)}</text>
				</g>
			{/each}

			<!-- axis labels -->
			<text
				x={innerW / 2}
				y={innerH + 34}
				text-anchor="middle"
				class="axis-label"
			>
				{xLabel}
			</text>
			<text
				transform="translate(-42, {innerH / 2}) rotate(-90)"
				text-anchor="middle"
				class="axis-label"
			>
				{yLabel}
			</text>

			<!-- points -->
			{#each points as [x, y], i}
				<circle
					cx={sx(x)}
					cy={sy(y)}
					r={pointRadius}
					class="pt"
					onmouseenter={() => onMouseEnter(i, x, y)}
					onmouseleave={onMouseLeave}
					role="button"
					tabindex="0"
				/>
			{/each}

			{#if hovered}
				<g transform="translate({sx(hovered.x)}, {sy(hovered.y)})">
					<circle r={5} class="pt-hover" />
					<text x={8} y={-6} class="tooltip">
						({fmt(hovered.x)}, {fmt(hovered.y)})
					</text>
				</g>
			{/if}
		</g>
	</svg>
</div>

<style>
	.plot-wrap {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.5rem;
		overflow: auto;
	}

	.axis {
		stroke: var(--border);
		stroke-width: 1;
	}

	.grid {
		stroke: var(--border);
		stroke-width: 1;
		opacity: 0.35;
	}

	.tick {
		fill: var(--text-dim);
		font-size: 11px;
		font-family: var(--mono);
	}

	.axis-label {
		fill: var(--text);
		font-size: 12px;
	}

	.pt {
		fill: var(--accent);
		opacity: 0.75;
		cursor: pointer;
	}

	.pt:hover {
		opacity: 1;
	}

	.pt-hover {
		fill: var(--accent);
		stroke: var(--text);
		stroke-width: 1.5;
	}

	.tooltip {
		fill: var(--text);
		font-size: 11px;
		font-family: var(--mono);
		paint-order: stroke;
		stroke: var(--bg);
		stroke-width: 3px;
		stroke-linejoin: round;
	}
</style>
