// ==========================================
// ????Vanilla JS ?詨?蝜芸?撘? (摰靽?)
// ==========================================
let view = { x: 0, y: 0, scale: 1 };
let drag = null;
let stateLabelMap = new Map();
const diagramBindings = new WeakSet();

const GRID = 10;
const CHANNEL_GAP = 18;
const CLEARANCE = 24;
const WIRE = "#71839a";
const INK = "#253244";
const DEFAULT_CANVAS_WIDTH = 1380;

const SOURCE_X0 = 78;
const SOURCE_RAIL_END_X = 320;
const RAIL_TAP_START_X = 360;
const LITERAL_TAP_GAP = 24;
const DEFAULT_GATE_X = 570;
const GATE_WIDTH = 82;
const OR_WIDTH = 96;
const FF_WIDTH = 150;
const FF_HEIGHT = 110;
const OUTPUT_ROUTE_BASE = 72;
const OUTPUT_ROUTE_GAP = 18;

const LEGEND_Y = 38;
const LEGEND_BOTTOM_Y = 126;
const DEFAULT_RAIL_BASE_Y = 210;
const RAIL_GAP = 42;
const DEFAULT_FF_TOP_Y = 190;
const FF_GAP = 190;
const DEFAULT_FEEDBACK_TOP_Y = 150;
const FEEDBACK_LEFT_X = 44;
const FEEDBACK_VERTICAL_GAP = 20;
const FEEDBACK_HORIZONTAL_GAP = 18;
const FEEDBACK_LEFT_GAP = 12;
const DEFAULT_LOGIC_MIN_Y = 224;
const TERM_VERTICAL_GAP = 26;
const NETWORK_VERTICAL_GAP = 34;

function resetDiagramView() {
  view = { x: 0, y: 0, scale: 1 };
}

function zoomDiagram(svg, delta) {
  view.scale = Math.min(2.2, Math.max(0.55, view.scale + delta));
  applyView(svg);
}

function bindDiagramPan(svg) {
  if (diagramBindings.has(svg)) return;
  diagramBindings.add(svg);
  svg.style.userSelect = "none";
  svg.style.webkitUserSelect = "none";
  svg.style.touchAction = "none";
  svg.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    drag = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    event.preventDefault();
    view.x += event.movementX / view.scale;
    view.y += event.movementY / view.scale;
    drag = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
    applyView(svg);
  });

  svg.addEventListener("pointerup", (event) => {
    event.preventDefault();
    drag = null;
  });

  svg.addEventListener("pointercancel", (event) => {
    event.preventDefault();
    drag = null;
  });
  
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const delta = Math.max(-0.05, Math.min(0.05, -event.deltaY * 0.001));
    zoomDiagram(svg, delta);
  });
}

function renderCircuitDiagram(svg, analysis) {
  if (!analysis) return;

  stateLabelMap = new Map(analysis.variables.state.map((name) => [name, formatStateBit(name)]));

  const sourcePlan = collectSourcePlan(analysis);
  const verticalLayout = makeVerticalLayout(sourcePlan);
  const sourceRows = buildSourceRows(sourcePlan, verticalLayout.railBaseY);
  const layout = { ...verticalLayout, ...makeHorizontalLayout(sourceRows, analysis.equations.length) };
  const ffPositions = flipFlopPositions(analysis, layout, verticalLayout);
  const provisionalFfMap = new Map(ffPositions.map((ff) => [ff.name, ff]));
  const outputY = snap(
    Math.max(
      verticalLayout.railBaseY + Math.max(0, sourceRows.size - 1) * RAIL_GAP + 170,
      verticalLayout.ffTopY + Math.max(1, ffPositions.length) * FF_GAP + 72
    )
  );
  const logicLanes = assignLogicLanes(analysis.equations, provisionalFfMap, outputY, layout, verticalLayout);
  const maxLogicY = Math.max(outputY, ...logicLanes.values());
  const clockY = snap(maxLogicY + 190);
  const height = Math.max(760, clockY + 70);

  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${layout.canvasWidth} ${height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");

  const root = createSvgElement("g", { id: "viewportRoot" });
  const wires = createSvgElement("g", { id: "wireLayer" });
  const components = createSvgElement("g", { id: "componentLayer" });
  const labels = createSvgElement("g", { id: "labelLayer" });
  const dots = createSvgElement("g", { id: "junctionLayer" });
  root.append(wires, components, labels, dots);
  svg.appendChild(root);

  const ctx = { wires, components, labels, dots, layout };

  drawEquationLegend(ctx, analysis.equations);
  drawSourceRails(ctx, sourceRows);
  const ffMap = drawFlipFlops(ctx, ffPositions);
  drawFeedbackWires(ctx, sourceRows, ffMap, analysis);
  drawEquations(ctx, analysis, sourceRows, ffMap, outputY, logicLanes);
  drawClock(ctx, ffMap, clockY);

  applyView(svg);
}

function drawEquationLegend(ctx, equations) {
  text(ctx.labels, 72, LEGEND_Y, "Simplified Equations", "start", 16, false);
  equations.forEach((equation, index) => {
    const x = 72 + (index % 3) * 350;
    const y = LEGEND_Y + 34 + Math.floor(index / 3) * 28;
    text(ctx.labels, x, y, `${formatEquationName(equation.name)} = ${formatExpression(equation.expression)}`, "start", 15, true);
  });
  line(ctx.wires, 56, LEGEND_BOTTOM_Y, ctx.layout.outputX + 16, LEGEND_BOTTOM_Y, "#d8e0ea", 1.2);
}

function drawSourceRails(ctx, sourceRows) {
  const xRail = sourceRows.get("X");
  const invertedX = sourceRows.get("X'");

  for (const source of sourceRows.values()) {
    const startX = source.name === "X'" ? SOURCE_RAIL_END_X : SOURCE_X0;
    if (source.railEndX > startX) {
      line(ctx.wires, startX, source.y, source.railEndX, source.y);
    }
    if (source.name === "X") {
      text(ctx.labels, SOURCE_X0 - 14, source.y, source.label, "end", 15, true);
    }
  }

  if (xRail && invertedX) {
    drawInputInverter(ctx, xRail, invertedX);
  }
}

function drawInputInverter(ctx, normal, inverted) {
  const tapX = SOURCE_X0 + 58;
  const gateX = SOURCE_X0 + 96;
  const gateHeight = 20;
  const gateCenterY = snap((normal.y + inverted.y) / 2);
  const gate = notGate(ctx.components, gateX, gateCenterY - gateHeight / 2, 28, gateHeight);

  junction(ctx.dots, tapX, normal.y);
  polyline(ctx.wires, [
    [tapX, normal.y],
    [tapX, gateCenterY],
    [gateX, gateCenterY],
  ]);
  polyline(ctx.wires, [
    [gate.outX, gate.outY],
    [gate.outX + 22, gate.outY],
    [gate.outX + 22, inverted.y],
    [SOURCE_RAIL_END_X, inverted.y],
  ]);
  text(ctx.labels, gate.outX + 34, inverted.y - 12, "X'", "start", 14, true);
}

function drawFeedbackWires(ctx, sourceRows, ffMap, analysis) {
  let feedbackIndex = 0;

  analysis.variables.state.forEach((qName) => {
    const ff = ffMap.get(qName);
    if (!ff) return;

    const normal = sourceRows.get(qName);
    if (normal) {
      routeFeedbackToRail(ctx, ff.outputs.normal, normal, feedbackIndex);
      feedbackIndex += 1;
    }

    const inverted = sourceRows.get(`${qName}'`);
    if (inverted) {
      routeFeedbackToRail(ctx, ff.outputs.inverted, inverted, feedbackIndex);
      feedbackIndex += 1;
    }
  });
}

function routeFeedbackToRail(ctx, output, source, feedbackIndex) {
  const turnX = snap(ctx.layout.feedbackRightX + feedbackIndex * FEEDBACK_VERTICAL_GAP);
  const trackY = snap(ctx.layout.feedbackTopY + feedbackIndex * FEEDBACK_HORIZONTAL_GAP);
  const leftX = snap(ctx.layout.feedbackLeftBaseX - feedbackIndex * FEEDBACK_LEFT_GAP);

  polyline(ctx.wires, [
    [output.x, output.y],
    [turnX, output.y],
    [turnX, trackY],
    [leftX, trackY],
    [leftX, source.y],
    [SOURCE_X0, source.y],
  ]);
}

function drawEquations(ctx, analysis, sourceRows, ffMap, outputY, logicLanes) {
  analysis.equations.forEach((equation, index) => {
    const target = getEquationTarget(equation, ffMap, outputY, ctx.layout);
    const logicY = logicLanes.get(equation.name) ?? target.y;
    const output = drawExpressionNetwork(ctx, equation.expression, sourceRows, logicY, index);
    routeOutputToTarget(ctx, output, target, index);

    if (target.kind === "output") {
      text(ctx.labels, target.x + 16, target.y, "Z", "start", 16, true);
    }
  });
}

function drawExpressionNetwork(ctx, expression, sourceRows, y, laneIndex) {
  if (expression === "0" || expression === "1") {
    const constant = constantNode(ctx.components, ctx.labels, ctx.layout.gateX, y - 16, expression);
    return { x: constant.outX, y: constant.outY };
  }

  const terms = sortTermsBySourceY(splitTerms(expression), sourceRows);
  if (terms.length === 1) {
    return drawProductTerm(ctx, terms[0], sourceRows, y, laneIndex, 0);
  }

  const termLayouts = layoutProductTerms(terms, y);
  const totalTermHeight = termLayouts.at(-1).bottom - termLayouts[0].top;
  const orGateHeight = Math.max(58, totalTermHeight + 18);
  const or = orGate(ctx.components, ctx.layout.orX, y - orGateHeight / 2, OR_WIDTH, orGateHeight, terms.length);
  termLayouts.forEach(({ term, centerY }, termIndex) => {
    const product = drawProductTerm(ctx, term, sourceRows, centerY, laneIndex, termIndex);
    const input = or.inputs[termIndex];
    const routeX = snap(ctx.layout.orX - CLEARANCE - termIndex * CHANNEL_GAP);
    polyline(ctx.wires, [
      [product.x, product.y],
      [routeX, product.y],
      [routeX, input.y],
      [input.x, input.y],
    ]);
  });

  return { x: or.outX, y: or.outY };
}

function drawProductTerm(ctx, term, sourceRows, y, laneIndex, termIndex) {
  if (term === "0" || term === "1") {
    const constant = constantNode(ctx.components, ctx.labels, ctx.layout.gateX, y - 16, term);
    return { x: constant.outX, y: constant.outY };
  }

  const literals = sortLiteralsBySourceY(parseLiterals(term), sourceRows);

  if (literals.length === 1) {
    const point = { x: ctx.layout.gateX + GATE_WIDTH - 6, y };
    routeLiteralToPoint(ctx, literals[0], sourceRows, point);
    return point;
  }

  const gateHeight = productTermHeight(term);
  const gate = andGate(ctx.components, ctx.layout.gateX, y - gateHeight / 2, GATE_WIDTH, gateHeight, literals.length);
  literals.forEach((literal, literalIndex) => {
    routeLiteralToPoint(ctx, literal, sourceRows, gate.inputs[literalIndex]);
  });
  return { x: gate.outX, y: gate.outY };
}

function routeLiteralToPoint(ctx, literal, sourceRows, point) {
  const source = sourceRows.get(normalizeLiteral(literal));
  if (!source) {
    console.warn(`Diagram source missing for literal ${literal}.`);
    return;
  }

  const useIndex = source.nextTapIndex;
  source.nextTapIndex += 1;

  const tapX = snap(source.tapStartX + useIndex * LITERAL_TAP_GAP);

  junction(ctx.dots, tapX, source.y);
  polyline(ctx.wires, [
    [tapX, source.y],
    [tapX, point.y],
    [point.x, point.y],
  ]);
}

function routeOutputToTarget(ctx, output, target, laneIndex) {
  if (Math.abs(output.y - target.y) < 1) {
    line(ctx.wires, output.x, output.y, target.x, target.y);
    return;
  }

  const routeX = snap(target.x - OUTPUT_ROUTE_BASE - laneIndex * OUTPUT_ROUTE_GAP);
  polyline(ctx.wires, [
    [output.x, output.y],
    [routeX, output.y],
    [routeX, target.y],
    [target.x, target.y],
  ]);
}

function drawFlipFlops(ctx, positions) {
  const map = new Map();
  positions.forEach((ff) => {
    flipFlop(ctx, ff.x, ff.y, ff);
    map.set(ff.name, ff);
  });
  return map;
}

function drawClock(ctx, ffMap, clockY) {
  line(ctx.wires, SOURCE_X0, clockY, ctx.layout.outputX, clockY);
  text(ctx.labels, SOURCE_X0 - 14, clockY, "CLK", "end", 15, true);
  text(ctx.labels, ctx.layout.outputX + 20, clockY, "CLK", "start", 15, true);

  const flipFlops = [...ffMap.values()];
  if (!flipFlops.length) return;

  const trunkX = snap(Math.min(...flipFlops.map((ff) => ff.x)) - 46);
  const topClockY = Math.min(...flipFlops.map((ff) => ff.clockY));
  polyline(ctx.wires, [
    [trunkX, clockY],
    [trunkX, topClockY],
  ]);
  junction(ctx.dots, trunkX, clockY);

  for (const ff of flipFlops) {
    const clockPinX = snap(ff.x + ff.width / 2);
    polyline(ctx.wires, [
      [trunkX, ff.clockY],
      [clockPinX, ff.clockY],
    ]);
    junction(ctx.dots, trunkX, ff.clockY);
  }
}

function collectSourcePlan(analysis) {
  const literalCounts = countActiveLiterals(analysis);
  const used = new Set(literalCounts.keys());
  const ordered = [];

  if (used.has("X") || used.has("X'")) ordered.push("X");
  if (used.has("X'")) ordered.push("X'");

  analysis.variables.state.forEach((name) => {
    if (used.has(name)) ordered.push(name);
    if (used.has(`${name}'`)) ordered.push(`${name}'`);
  });

  return {
    ordered,
    literalCounts,
    feedbackCount: ordered.filter((name) => name.startsWith("Q")).length,
  };
}

function makeVerticalLayout(sourcePlan) {
  const feedbackTrackCount = Math.max(0, sourcePlan.feedbackCount);
  const feedbackTopY = DEFAULT_FEEDBACK_TOP_Y;
  const feedbackBottomY = feedbackTrackCount > 0 ? feedbackTopY + (feedbackTrackCount - 1) * FEEDBACK_HORIZONTAL_GAP : LEGEND_BOTTOM_Y;
  const railBaseY = snap(Math.max(DEFAULT_RAIL_BASE_Y, feedbackBottomY + 56));
  return {
    feedbackTopY,
    feedbackLeftBaseX: snap(20 + Math.max(0, feedbackTrackCount - 1) * FEEDBACK_LEFT_GAP),
    railBaseY,
    ffTopY: snap(Math.max(DEFAULT_FF_TOP_Y, railBaseY - 20)),
    logicMinY: snap(Math.max(DEFAULT_LOGIC_MIN_Y, railBaseY + 14)),
  };
}

function buildSourceRows(sourcePlan, railBaseY) {
  const planned = sourcePlan.ordered.map((name, index) => ({
    name,
    index,
    label: formatSourceLabel(name),
    y: snap(railBaseY + index * RAIL_GAP),
    tapCount: sourcePlan.literalCounts.get(name) ?? 0,
    nextTapIndex: 0,
  }));

  let laneCursor = 0;
  [...planned].reverse().forEach((source) => {
    if (source.tapCount > 0) {
      source.tapStartX = snap(RAIL_TAP_START_X + laneCursor * LITERAL_TAP_GAP);
      source.tapEndX = snap(source.tapStartX + (source.tapCount - 1) * LITERAL_TAP_GAP);
      laneCursor += source.tapCount;
    } else {
      source.tapStartX = SOURCE_RAIL_END_X;
      source.tapEndX = SOURCE_RAIL_END_X;
    }
    source.railEndX = Math.max(SOURCE_RAIL_END_X, source.tapEndX);
  });

  return new Map(
    planned.map((source) => [source.name, source])
  );
}

function countActiveLiterals(analysis) {
  const counts = new Map();
  analysis.equations.forEach((equation) => {
    parseLiterals(equation.expression).forEach((literal) => {
      counts.set(literal, (counts.get(literal) ?? 0) + 1);
    });
  });
  return counts;
}

function makeHorizontalLayout(sourceRows, equationCount) {
  const maxRailEndX = Math.max(SOURCE_RAIL_END_X, ...[...sourceRows.values()].map((source) => source.railEndX));
  const gateX = snap(Math.max(DEFAULT_GATE_X, maxRailEndX + 110));
  const orX = snap(gateX + GATE_WIDTH + 120);
  const outputRouteWidth = OUTPUT_ROUTE_BASE + Math.max(0, equationCount - 1) * OUTPUT_ROUTE_GAP;
  const ffX = snap(orX + OR_WIDTH + outputRouteWidth + 132);
  const outputX = snap(ffX + FF_WIDTH + 130);
  const feedbackRightX = snap(outputX + 84);
  const canvasWidth = snap(Math.max(DEFAULT_CANVAS_WIDTH, feedbackRightX + 120));

  return {
    gateX,
    orX,
    ffX,
    outputX,
    feedbackRightX,
    canvasWidth,
  };
}

function assignLogicLanes(equations, ffMap, outputY, layout, verticalLayout) {
  const sorted = equations
    .map((equation, index) => ({
      equation,
      index,
      targetY: getEquationTarget(equation, ffMap, outputY, layout).y,
      blockHeight: expressionBlockHeight(equation.expression),
    }))
    .sort((a, b) => a.targetY - b.targetY || a.index - b.index);

  const lanes = new Map();
  let previousBottom = verticalLayout.logicMinY - NETWORK_VERTICAL_GAP;

  sorted.forEach(({ equation, targetY, blockHeight }) => {
    const halfHeight = blockHeight / 2;
    const laneY = snap(Math.max(targetY, previousBottom + NETWORK_VERTICAL_GAP + halfHeight));
    lanes.set(equation.name, laneY);
    previousBottom = laneY + halfHeight;
  });

  return lanes;
}

function getEquationTarget(equation, ffMap, outputY, layout) {
  if (equation.type === "output") {
    return { kind: "output", x: layout.outputX, y: outputY };
  }

  const ffName = equation.name.match(/Q\d+$/)?.[0];
  const ff = ffMap.get(ffName);
  if (!ff) throw new Error(`Diagram target missing for ${equation.name}.`);

  const prefix = equation.name[0];
  if (prefix === "K" || prefix === "R") return { kind: "ff", x: ff.x, y: ff.inputs.lower.y };
  if (prefix === "T" || prefix === "D") return { kind: "ff", x: ff.x, y: ff.inputs.single.y };
  return { kind: "ff", x: ff.x, y: ff.inputs.upper.y };
}

function flipFlopPositions(analysis, layout, verticalLayout) {
  return analysis.graph.flipFlops.map((ff, index) => {
    const x = layout.ffX;
    const y = verticalLayout.ffTopY + index * FF_GAP;
    const width = FF_WIDTH;
    const height = FF_HEIGHT;
    const label = formatStateName(ff.name);
    const hasTwoInputs = ff.type === "JK" || ff.type === "SR";
    return {
      ...ff,
      label,
      x,
      y,
      width,
      height,
      inputs: hasTwoInputs
        ? {
            upper: { x: snap(x), y: snap(y + 34) },
            lower: { x: snap(x), y: snap(y + 72) },
          }
        : {
            single: { x: snap(x), y: snap(y + 54) },
          },
      outputs: {
        normal: { x: snap(x + width + 42), y: snap(y + 36) },
        inverted: { x: snap(x + width + 42), y: snap(y + 74) },
      },
      clockY: snap(y + height),
    };
  });
}

function flipFlop(ctx, x, y, ff) {
  rect(ctx.components, x, y, ff.width, ff.height, "#ffffff", WIRE, 1.8);

  if (ff.type === "JK" || ff.type === "SR") {
    const [upperPrefix, lowerPrefix] = ff.type === "JK" ? ["J", "K"] : ["S", "R"];
    text(ctx.labels, x + 18, ff.inputs.upper.y, `${upperPrefix}${ff.label}`, "start", 14, true);
    text(ctx.labels, x + 18, ff.inputs.lower.y, `${lowerPrefix}${ff.label}`, "start", 14, true);
  } else {
    const prefix = ff.type === "D" ? "D" : "T";
    text(ctx.labels, x + 18, ff.inputs.single.y, `${prefix}${ff.label}`, "start", 14, true);
  }

  line(ctx.wires, x + ff.width, ff.outputs.normal.y, ff.outputs.normal.x, ff.outputs.normal.y);
  line(ctx.wires, x + ff.width, ff.outputs.inverted.y, ff.outputs.inverted.x, ff.outputs.inverted.y);
  text(ctx.labels, x + ff.width - 14, ff.outputs.normal.y, ff.label, "end", 14, true);
  text(ctx.labels, x + ff.width - 14, ff.outputs.inverted.y, `${ff.label}'`, "end", 14, true);

  path(ctx.components, `M ${x + 65} ${ff.clockY} L ${x + 75} ${ff.clockY - 12} L ${x + 85} ${ff.clockY}`, WIRE, "none", 1.5);
  text(ctx.labels, x + 75, ff.clockY + 14, "CLK", "middle", 11, false);
}

function andGate(components, x, y, width, height, inputCount) {
  const d = [
    `M ${x} ${y}`,
    `L ${x + width / 2} ${y}`,
    `C ${x + width} ${y}, ${x + width} ${y + height}, ${x + width / 2} ${y + height}`,
    `L ${x} ${y + height}`,
    "Z",
  ].join(" ");
  path(components, d, WIRE, "#ffffff", 1.7);
  return {
    inputs: inputPins(x, y, height, inputCount),
    outX: snap(x + width),
    outY: snap(y + height / 2),
  };
}

function orGate(components, x, y, width, height, inputCount) {
  const d = [
    `M ${x} ${y}`,
    `C ${x + 28} ${y + 10}, ${x + 28} ${y + height - 10}, ${x} ${y + height}`,
    `C ${x + 44} ${y + height - 8}, ${x + width - 22} ${y + height - 5}, ${x + width} ${y + height / 2}`,
    `C ${x + width - 22} ${y + 5}, ${x + 44} ${y + 8}, ${x} ${y}`,
  ].join(" ");
  path(components, d, WIRE, "#ffffff", 1.8);
  return {
    inputs: inputPins(x + 5, y, height, inputCount),
    outX: snap(x + width),
    outY: snap(y + height / 2),
  };
}

function notGate(components, x, y, width, height) {
  path(components, `M ${x} ${y} L ${x} ${y + height} L ${x + width} ${y + height / 2} Z`, WIRE, "#ffffff", 1.6);
  circle(components, x + width + 6, y + height / 2, 4, "#ffffff", WIRE, 1.5);
  return { outX: snap(x + width + 10), outY: snap(y + height / 2) };
}

function constantNode(components, labels, x, y, value) {
  const label = value === "1" ? "VCC" : "GND";
  rect(components, x, y, 60, 32, "#ffffff", WIRE, 1.5);
  text(labels, x + 30, y + 16, label, "middle", 14, true);
  return { outX: snap(x + 60), outY: snap(y + 16) };
}

function inputPins(x, y, height, count) {
  return Array.from({ length: count }, (_, index) => ({
    x: snap(x),
    y: snap(y + ((index + 1) * height) / (count + 1)),
  }));
}

function splitTerms(expression) {
  if (expression === "0" || expression === "1") return [expression];
  return expression.split("+").map((term) => term.trim()).filter(Boolean);
}

function parseLiterals(term) {
  if (term === "0" || term === "1") return [];
  return (term.match(/Q(?:\d|[\u2080-\u2089])+'?|X'?/g) ?? []).map(normalizeLiteral);
}

function layoutProductTerms(terms, centerY) {
  const items = terms.map((term) => ({ term, height: productTermHeight(term) }));
  const totalHeight =
    items.reduce((sum, item) => sum + item.height, 0) + Math.max(0, items.length - 1) * TERM_VERTICAL_GAP;
  let cursor = centerY - totalHeight / 2;

  return items.map((item) => {
    const top = cursor;
    const center = top + item.height / 2;
    cursor += item.height + TERM_VERTICAL_GAP;
    return {
      term: item.term,
      top,
      bottom: top + item.height,
      centerY: snap(center),
    };
  });
}

function expressionBlockHeight(expression) {
  const terms = splitTerms(expression);
  if (terms.length <= 1) return productTermHeight(terms[0] ?? expression);
  return terms.reduce((sum, term) => sum + productTermHeight(term), 0) + (terms.length - 1) * TERM_VERTICAL_GAP;
}

function productTermHeight(term) {
  if (term === "0" || term === "1") return 32;
  const literalCount = parseLiterals(term).length;
  if (literalCount <= 1) return 36;
  return Math.max(54, literalCount * 24 + 18);
}

function sortTermsBySourceY(terms, sourceRows) {
  return [...terms].sort((a, b) => termSourceY(a, sourceRows) - termSourceY(b, sourceRows));
}

function sortLiteralsBySourceY(literals, sourceRows) {
  return [...literals].sort((a, b) => sourceY(a, sourceRows) - sourceY(b, sourceRows));
}

function termSourceY(term, sourceRows) {
  const literals = parseLiterals(term);
  if (!literals.length) return Number.POSITIVE_INFINITY;
  return Math.min(...literals.map((literal) => sourceY(literal, sourceRows)));
}

function sourceY(literal, sourceRows) {
  return sourceRows.get(normalizeLiteral(literal))?.y ?? Number.POSITIVE_INFINITY;
}

function formatEquationName(name) {
  return name.replace(/Q(?:\d|[\u2080-\u2089])+/g, (qName) => formatStateName(normalizeLiteral(qName)));
}

function formatExpression(expression) {
  return expression.replace(/Q(?:\d|[\u2080-\u2089])+/g, (qName) => formatStateName(normalizeLiteral(qName))).replace(/\s+/g, "");
}

function formatSourceLabel(literal) {
  const inverted = literal.endsWith("'");
  const base = inverted ? literal.slice(0, -1) : literal;
  if (!base.startsWith("Q")) return `${base}${inverted ? "'" : ""}`;
  return `${formatStateName(base)}${inverted ? "'" : ""}`;
}

function isExternalInputSignal(name) {
  return name === "X" || name === "X'";
}

function formatStateName(name) {
  const normalized = normalizeLiteral(name);
  return stateLabelMap.get(normalized) ?? formatStateBit(normalized);
}

const SUBSCRIPT_DIGITS = {
  0: "\u2080", 1: "\u2081", 2: "\u2082", 3: "\u2083", 4: "\u2084",
  5: "\u2085", 6: "\u2086", 7: "\u2087", 8: "\u2088", 9: "\u2089",
};

const NORMAL_DIGITS = Object.fromEntries(Object.entries(SUBSCRIPT_DIGITS).map(([digit, subscript]) => [subscript, digit]));

function toSubscript(str) {
  return String(str).replace(/[0-9]/g, (match) => SUBSCRIPT_DIGITS[match]);
}

function fromSubscript(str) {
  return String(str).replace(/[\u2080-\u2089]/g, (match) => NORMAL_DIGITS[match]);
}

function normalizeLiteral(literal) {
  return fromSubscript(literal);
}

function formatStateBit(qName) {
  return String(qName ?? "").replace(/^Q(\d+)$/, (_, digits) => `Q${toSubscript(digits)}`);
}

function snap(value) {
  return Math.round(value / GRID) * GRID;
}

function polyline(root, points, stroke = WIRE, width = 1.5) {
  const element = createSvgElement("polyline", {
    points: points.map(([x, y]) => `${snap(x)},${snap(y)}`).join(" "),
    fill: "none",
    stroke,
    "stroke-width": width,
    "stroke-linejoin": "round",
    "stroke-linecap": "round",
  });
  root.appendChild(element);
}

function line(root, x1, y1, x2, y2, stroke = WIRE, width = 1.5) {
  root.appendChild(
    createSvgElement("line", {
      x1: snap(x1), y1: snap(y1), x2: snap(x2), y2: snap(y2),
      stroke, "stroke-width": width, "stroke-linecap": "round",
    })
  );
}

function rect(root, x, y, width, height, fill, stroke, strokeWidth) {
  root.appendChild(createSvgElement("rect", { x: snap(x), y: snap(y), width, height, fill, stroke, "stroke-width": strokeWidth }));
}

function circle(root, cx, cy, r, fill, stroke, strokeWidth) {
  root.appendChild(createSvgElement("circle", { cx: snap(cx), cy: snap(cy), r, fill, stroke, "stroke-width": strokeWidth }));
}

function junction(root, cx, cy) {
  circle(root, cx, cy, 3, INK, INK, 1);
}

function path(root, d, stroke, fill, strokeWidth) {
  root.appendChild(
    createSvgElement("path", {
      d, fill, stroke, "stroke-width": strokeWidth,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    })
  );
}

function text(root, x, y, value, anchor = "start", size = 14, italic = false) {
  const node = createSvgElement("text", {
    x: snap(x), y: snap(y),
    "text-anchor": anchor,
    "dominant-baseline": "middle",
    fill: INK,
    "font-size": size,
    "font-weight": italic ? 700 : 500,
    "font-family": italic ? "Georgia, Times New Roman, serif" : "Segoe UI, sans-serif",
    "font-style": italic ? "italic" : "normal",
  });
  node.textContent = value;
  root.appendChild(node);
}

function createSvgElement(tag, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function applyView(svg) {
  const root = svg.querySelector("#viewportRoot");
  if (!root) return;
  root.setAttribute("transform", `translate(${view.x} ${view.y}) scale(${view.scale})`);
}

// ==========================================
// React ?辣???蝔?頧??// ==========================================

export { resetDiagramView, renderCircuitDiagram, bindDiagramPan };

