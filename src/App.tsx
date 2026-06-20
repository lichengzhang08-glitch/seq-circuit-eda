import { useEffect, useMemo, useRef, useState } from "react";
import { bindDiagramPan, renderCircuitDiagram, resetDiagramView } from "./diagramEngine";

type ModelType = "mealy" | "moore";
type FlipFlopType = "jk" | "d" | "t";
type TriggerEdge = "rising" | "falling";

type StateRow = {
  presentState: string;
  input: string;
  nextState: string;
  output: string;
};

type EquationResult = {
  signal: string;
  equation: string;
  minterms: number[];
  dontCares: number[];
  implicants: string[];
  verified: boolean;
};

type AnalysisResult = {
  valid: boolean;
  errors: string[];
  states: string[];
  bitCount: number;
  stateBits: string[];
  stateAssignments: Record<string, string>;
  equations: EquationResult[];
  outputEquation: EquationResult;
  variableNames: string[];
};

const parserSchema = {
  type: "object",
  additionalProperties: false,
  required: ["modelType", "initialState", "states", "rows", "notes"],
  properties: {
    modelType: { type: "string", enum: ["mealy", "moore"] },
    initialState: { type: "string" },
    states: { type: "array", items: { type: "string" } },
    rows: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["state", "next0", "out0", "next1", "out1", "output"],
        properties: {
          state: { type: "string" },
          next0: { type: "string" },
          out0: { type: "string", enum: ["0", "1"] },
          next1: { type: "string" },
          out1: { type: "string", enum: ["0", "1"] },
          output: { type: "string", enum: ["0", "1"] },
        },
      },
    },
    notes: { type: "string" },
  },
};

const exampleRows: StateRow[] = [
  { presentState: "00", input: "0", nextState: "00", output: "0" },
  { presentState: "00", input: "1", nextState: "01", output: "0" },
  { presentState: "01", input: "0", nextState: "01", output: "0" },
  { presentState: "01", input: "1", nextState: "10", output: "0" },
  { presentState: "10", input: "0", nextState: "10", output: "0" },
  { presentState: "10", input: "1", nextState: "11", output: "0" },
  { presentState: "11", input: "0", nextState: "11", output: "0" },
  { presentState: "11", input: "1", nextState: "00", output: "1" },
];

const detectorRows: StateRow[] = [
  { presentState: "A", input: "0", nextState: "A", output: "0" },
  { presentState: "A", input: "1", nextState: "B", output: "0" },
  { presentState: "B", input: "0", nextState: "A", output: "0" },
  { presentState: "B", input: "1", nextState: "C", output: "0" },
  { presentState: "C", input: "0", nextState: "A", output: "0" },
  { presentState: "C", input: "1", nextState: "D", output: "1" },
  { presentState: "D", input: "0", nextState: "A", output: "0" },
  { presentState: "D", input: "1", nextState: "D", output: "1" },
];

const toggleRows: StateRow[] = [
  { presentState: "A", input: "0", nextState: "A", output: "0" },
  { presentState: "A", input: "1", nextState: "B", output: "1" },
  { presentState: "B", input: "0", nextState: "B", output: "1" },
  { presentState: "B", input: "1", nextState: "A", output: "0" },
];

const blankRow: StateRow = {
  presentState: "",
  input: "",
  nextState: "",
  output: "",
};

async function parseStateTableWithOpenAI({
  apiKey,
  model,
  problemText,
  modelType,
}: {
  apiKey: string;
  model: string;
  problemText: string;
  modelType: ModelType;
}) {
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: buildSystemPrompt() },
        {
          role: "user",
          content: `Selected model type: ${modelType}\n\nAssume exactly one binary input variable X and one binary output variable Z.\n\nConvert this sequential-circuit problem into the required normalized state table JSON:\n\n${problemText}`,
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "state_table_parse",
          strict: true,
          schema: parserSchema,
        },
      },
    }),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(payload?.error?.message ?? `API Error ${response.status}`);
  const text = payload?.choices?.[0]?.message?.content;
  if (!text) throw new Error("No JSON text returned.");
  return normalizeAiRows(JSON.parse(text), modelType);
}

function buildSystemPrompt() {
  return "You are a digital logic design assistant. Convert natural-language sequential circuit descriptions into a normalized state table. Only support one binary input X and one binary output Z. Use the user's selected model type exactly: mealy or moore. For Mealy machines, output may depend on present state and input. Fill out0 and out1. Set output to 0. For Moore machines, output depends only on present state. Fill output. Also set out0 and out1 equal to output. Create enough states to represent the required history for overlapping sequence detection when needed. Every state must define next0 and next1. Use short state labels A, B, C, D unless the problem clearly provides labels. Do not simplify equations and do not draw circuits.";
}

function normalizeAiRows(parsed: any, selectedModelType: ModelType) {
  const rows = parsed.rows.map((row: any) =>
    selectedModelType === "mealy"
      ? { state: row.state, next0: row.next0, out0: row.out0, next1: row.next1, out1: row.out1 }
      : { state: row.state, output: row.output, next0: row.next0, next1: row.next1 },
  );
  return { rows, states: parsed.states as string[] };
}

function adaptAiRowsToStateRows(parsed: { rows: any[]; states: string[] }, selectedModelType: ModelType): StateRow[] {
  const allStates = Array.from(
    new Set(
      [
        ...parsed.states,
        ...parsed.rows.flatMap((row) => [row.state, row.next0, row.next1]),
      ]
        .filter(Boolean)
        .map((state) => String(state).toUpperCase()),
    ),
  );
  const bitCount = Math.max(1, Math.ceil(Math.log2(Math.max(2, allStates.length))));
  const assignment = Object.fromEntries(
    allStates.map((state, index) => [state, /^[01]+$/.test(state) ? state.padStart(bitCount, "0") : index.toString(2).padStart(bitCount, "0")]),
  );
  return parsed.rows.flatMap((row) => {
    const presentState = assignment[String(row.state).toUpperCase()];
    const next0 = assignment[String(row.next0).toUpperCase()];
    const next1 = assignment[String(row.next1).toUpperCase()];
    if (selectedModelType === "mealy") {
      return [
        { presentState, input: "0", nextState: next0, output: row.out0 },
        { presentState, input: "1", nextState: next1, output: row.out1 },
      ];
    }
    return [
      { presentState, input: "0", nextState: next0, output: row.output },
      { presentState, input: "1", nextState: next1, output: row.output },
    ];
  });
}

function App() {
  const [modelType, setModelType] = useState<ModelType>("mealy");
  const [flipFlopType, setFlipFlopType] = useState<FlipFlopType>("jk");
  const [triggerEdge, setTriggerEdge] = useState<TriggerEdge>("rising");
  const [inputVariable, setInputVariable] = useState("X");
  const [outputVariable, setOutputVariable] = useState("Z");
  const [description, setDescription] = useState("");
  const [openAiApiKey, setOpenAiApiKey] = useState("");
  const [openAiModel, setOpenAiModel] = useState("gpt-4o-mini");
  const [isParsingText, setIsParsingText] = useState(false);
  const [stateRows, setStateRows] = useState<StateRow[]>(exampleRows);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [selectedSignal, setSelectedSignal] = useState("");
  const [generationStatus, setGenerationStatus] = useState("Ready");
  const [csvText, setCsvText] = useState("");
  const [leftPaneWidth, setLeftPaneWidth] = useState(30);
  const resizingPaneRef = useRef(false);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!resizingPaneRef.current) return;
      const nextWidth = Math.min(48, Math.max(24, (event.clientX / window.innerWidth) * 100));
      setLeftPaneWidth(nextWidth);
    };
    const handlePointerUp = () => {
      resizingPaneRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, []);

  const startPaneResize = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    resizingPaneRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  const liveAnalysis = useMemo(
    () =>
      analyzeCircuit(
        stateRows,
        flipFlopType,
        inputVariable.trim().toUpperCase() || "X",
        outputVariable.trim().toUpperCase() || "Z",
      ),
    [stateRows, flipFlopType, inputVariable, outputVariable],
  );

  const outputAnalysis = liveAnalysis;

  const selectedEquation = useMemo(() => {
    if (!outputAnalysis) return null;
    return (
      outputAnalysis.equations.find((equation) => equation.signal === selectedSignal) ??
      outputAnalysis.equations[0] ??
      outputAnalysis.outputEquation
    );
  }, [outputAnalysis, selectedSignal]);

  const updateRow = (
    index: number,
    field: keyof StateRow,
    value: string,
  ) => {
    setStateRows((rows) =>
      rows.map((row, rowIndex) =>
        rowIndex === index ? { ...row, [field]: value.toUpperCase() } : row,
      ),
    );
  };

  const addRow = () => {
    setStateRows((rows) => [...rows, blankRow]);
  };

  const clearRows = () => {
    setStateRows([blankRow]);
    setAnalysis(null);
    setGenerationStatus("Table cleared");
  };

  const loadExample = () => {
    setModelType("mealy");
    setFlipFlopType("jk");
    setStateRows(exampleRows);
    setAnalysis(null);
    setGenerationStatus("2-bit up-counter example loaded");
  };

  const loadNamedExample = (name: string) => {
    if (name === "detector") {
      setModelType("mealy");
      setFlipFlopType("d");
      setStateRows(detectorRows);
      setGenerationStatus("Sequence detector example loaded");
    }
    if (name === "toggle") {
      setModelType("mealy");
      setFlipFlopType("t");
      setStateRows(toggleRows);
      setGenerationStatus("Toggle controller example loaded");
    }
    if (name === "basic") {
      loadExample();
      return;
    }
    setAnalysis(null);
  };

  const parseDescription = async () => {
    if (!description.trim()) {
      setGenerationStatus("Please enter a natural language problem first.");
      return;
    }
    if (!openAiApiKey.trim()) {
      setGenerationStatus("OpenAI API key is required for AI parsing.");
      return;
    }
    setIsParsingText(true);
    setGenerationStatus("Parsing text with OpenAI...");
    try {
      const parsed = await parseStateTableWithOpenAI({
        apiKey: openAiApiKey.trim(),
        model: openAiModel.trim() || "gpt-4o-mini",
        problemText: description,
        modelType,
      });
      const convertedRows = adaptAiRowsToStateRows(parsed, modelType);
      setStateRows(convertedRows);
      setInputVariable("X");
      setOutputVariable("Z");
      setAnalysis(null);
      setSelectedSignal("");
      setGenerationStatus("OpenAI parsed the problem into a binary state table");
    } catch (err: any) {
      console.error("API Error:", err);
      let userFriendlyMessage = "文字解析失敗，請確認網路連線或 API Key 格式。";
      if (err.message && err.message.includes("429")) {
        userFriendlyMessage = "OpenAI API 請求失敗 (Error 429)：\n這把 API Key 的帳號免費額度已耗盡或未綁定信用卡儲值。\n請更換一組有餘額的 API Key！";
      } else if (err.message && err.message.includes("401")) {
        userFriendlyMessage = "OpenAI API 驗證失敗 (Error 401)：\n請檢查你的 API Key 是否複製完整，且不包含前後空白。";
      }
      alert(userFriendlyMessage);
    } finally {
      setIsParsingText(false);
    }
  };

  const handleGenerate = () => {
    const result = analyzeCircuit(
      stateRows,
      flipFlopType,
      inputVariable.trim().toUpperCase() || "X",
      outputVariable.trim().toUpperCase() || "Z",
    );
    setAnalysis(result);
    setSelectedSignal(result.equations[0]?.signal ?? "");
    setGenerationStatus(result.valid ? "Generated successfully" : "Validation failed");
  };

  const exportReport = () => {
    if (!outputAnalysis) return;
    const report = [
      "Sequential Circuit Design Automation System",
      "Student ID: 1140518",
      "Name: Chang Li Cheng",
      `Model Type: ${modelType}`,
      `Flip-Flop Type: ${flipFlopType.toUpperCase()}`,
      `Input Variable: ${inputVariable}`,
      `Output Variable: ${outputVariable}`,
      "",
      "State Assignments",
      ...Object.entries(outputAnalysis.stateAssignments).map(
        ([state, bits]) => `${state} = ${bits}`,
      ),
      "",
      "Equations",
      ...outputAnalysis.equations.map(
        (equation) => `${equation.signal} = ${equation.equation}`,
      ),
      `${outputAnalysis.outputEquation.signal} = ${outputAnalysis.outputEquation.equation}`,
    ].join("\n");
    const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "seqcircuit-report.txt";
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportCsv = () => {
    const csv = [
      "Present State,Input,Next State,Output",
      ...stateRows.map((row) =>
        [row.presentState, row.input, row.nextState, row.output].join(","),
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "state-table.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  const importCsv = () => {
    const rows = csvText
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .filter((line) => !line.toLowerCase().startsWith("present"))
      .map((line) => {
        const [presentState = "", input = "", nextState = "", output = ""] =
          line.split(",").map((cell) => cell.trim().toUpperCase());
        return { presentState, input, nextState, output };
      });
    if (!rows.length) {
      setGenerationStatus("CSV import failed: no valid rows found");
      return;
    }
    setStateRows(rows);
    setAnalysis(null);
    setGenerationStatus("CSV imported");
  };

  const copyEquations = async () => {
    if (!outputAnalysis) return;
    const text = [
      ...outputAnalysis.equations.map(
        (equation) => `${equation.signal} = ${equation.equation}`,
      ),
      `${outputAnalysis.outputEquation.signal} = ${outputAnalysis.outputEquation.equation}`,
    ].join("\n");
    await navigator.clipboard.writeText(text);
    setGenerationStatus("Equations copied to clipboard");
  };

  return (
    <main className="min-h-screen bg-[#f5f5f5] text-neutral-950">
      <div className="flex min-h-screen flex-col">
        <header className="sticky top-0 z-20 border-b border-neutral-200 bg-white/80 px-5 py-3 backdrop-blur-xl">
          <div className="mx-auto flex max-w-[1920px] items-center justify-between gap-5">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-neutral-500">
                SeqCircuit AI
              </p>
              <h1 className="mt-1 text-lg font-semibold tracking-tight text-neutral-950 md:text-xl">
                Sequential Circuit Design Automation System
              </h1>
            </div>
            <div className="rounded-[4px] border border-neutral-300 bg-neutral-950 px-4 py-2 text-right text-sm font-semibold leading-6 text-white">
              <div>1140518</div>
              <div>{"\u5f35\u7acb\u6f84"}</div>
            </div>
          </div>
        </header>

        <section className="mx-auto flex w-full max-w-[1920px] flex-1 gap-0 p-4 xl:h-[calc(100vh-148px)]">
          <aside
            className="flex min-h-0 min-w-[320px] flex-col gap-4 overflow-y-auto pr-4"
            style={{ width: `${leftPaneWidth}%` }}
          >
            <Panel title="1. Model Type">
              <div className="grid grid-cols-2 gap-3">
                <ChoiceCard
                  name="modelType"
                  label="Mealy"
                  description="Z depends on state and X"
                  checked={modelType === "mealy"}
                  onChange={() => setModelType("mealy")}
                />
                <ChoiceCard
                  name="modelType"
                  label="Moore"
                  description="Z depends on state only"
                  checked={modelType === "moore"}
                  onChange={() => setModelType("moore")}
                />
              </div>
            </Panel>

            <Panel title="2. Flip-Flop Type">
              <div className="grid grid-cols-3 gap-3">
                <ChoiceCard
                  name="flipFlopType"
                  label="JK"
                  description="J/K inputs"
                  checked={flipFlopType === "jk"}
                  onChange={() => setFlipFlopType("jk")}
                />
                <ChoiceCard
                  name="flipFlopType"
                  label="D-FF"
                  description="D inputs"
                  checked={flipFlopType === "d"}
                  onChange={() => setFlipFlopType("d")}
                />
                <ChoiceCard
                  name="flipFlopType"
                  label="T-FF"
                  description="T inputs"
                  checked={flipFlopType === "t"}
                  onChange={() => setFlipFlopType("t")}
                />
              </div>
              <div className="mt-4 rounded-[4px] border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm text-neutral-700">
                Current mode:{" "}
                <strong>{modelType === "mealy" ? "Mealy" : "Moore"}</strong>{" "}
                with{" "}
                <strong>
                  {flipFlopType === "jk"
                    ? "JK"
                    : flipFlopType === "d"
                      ? "D-FF"
                      : "T-FF"}
                </strong>
              </div>
            </Panel>

            <Panel title="3. Trigger Edge">
              <EdgeTriggerControl value={triggerEdge} onChange={setTriggerEdge} />
            </Panel>

            <Panel title="4. Natural Language Logic Input">
                <div className="mb-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                  <label className="block">
                    <span className="mb-1 block text-xs font-black uppercase tracking-wide text-slate-600">
                      OpenAI API Key
                    </span>
                    <input
                      type="password"
                      value={openAiApiKey}
                      onChange={(event) => setOpenAiApiKey(event.target.value)}
                      placeholder="sk-..."
                      className="h-10 w-full rounded-[3px] border border-neutral-300 bg-white px-3 text-sm font-medium text-neutral-950 outline-none transition focus:border-neutral-950"
                    />
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-xs font-black uppercase tracking-wide text-slate-600">
                      Model
                    </span>
                    <input
                      value={openAiModel}
                      onChange={(event) => setOpenAiModel(event.target.value)}
                      className="h-10 w-full rounded-[3px] border border-neutral-300 bg-white px-3 text-sm font-medium text-neutral-950 outline-none transition focus:border-neutral-950"
                    />
                  </label>
                </div>
                <textarea
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="Example: A Mealy system outputs 1 after the input has been 1 for three or more consecutive clock times."
                  className="min-h-28 w-full resize-y rounded-[3px] border border-neutral-300 bg-white px-3 py-3 text-sm leading-6 text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-neutral-950"
                />
                <button
                  type="button"
                  onClick={parseDescription}
                  disabled={isParsingText}
                  className="mt-3 h-10 w-full rounded-[3px] border border-neutral-950 bg-neutral-950 text-sm font-semibold text-white transition hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isParsingText ? "Parsing..." : "Parse Text to State Table"}
                </button>
              </Panel>

              <Panel title="5. State Table Input">
                <div className="mb-4 grid grid-cols-2 gap-3">
                  <label className="block">
                    <span className="mb-1 block text-xs font-black uppercase tracking-wide text-slate-600">
                      Input Variables
                    </span>
                    <input
                      value={inputVariable}
                      onChange={(event) =>
                        setInputVariable(event.target.value.toUpperCase())
                      }
                      className="h-10 w-full rounded-[3px] border border-neutral-300 bg-white px-3 text-sm font-medium text-neutral-950 outline-none transition focus:border-neutral-950"
                    />
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-xs font-black uppercase tracking-wide text-slate-600">
                      Output Variables
                    </span>
                    <input
                      value={outputVariable}
                      onChange={(event) =>
                        setOutputVariable(event.target.value.toUpperCase())
                      }
                      className="h-10 w-full rounded-[3px] border border-neutral-300 bg-white px-3 text-sm font-medium text-neutral-950 outline-none transition focus:border-neutral-950"
                    />
                  </label>
                </div>
                <div className="mb-3 grid grid-cols-3 gap-2">
                  <SmallButton onClick={addRow}>Add Row</SmallButton>
                  <SmallButton onClick={clearRows}>Clear Table</SmallButton>
                  <SmallButton onClick={loadExample}>Load Example</SmallButton>
                </div>
                <select
                  defaultValue=""
                  onChange={(event) => {
                    if (event.target.value) loadNamedExample(event.target.value);
                    event.target.value = "";
                  }}
                  className="mb-3 h-10 w-full rounded-[3px] border border-neutral-300 bg-white px-3 text-sm font-semibold text-neutral-700 outline-none transition focus:border-neutral-950"
                >
                  <option value="" disabled>
                    Quick Examples
                  </option>
                  <option value="basic">2-bit up-counter with enable</option>
                  <option value="detector">Three consecutive 1 detector</option>
                  <option value="toggle">Toggle controller</option>
                </select>
                <div className="overflow-hidden rounded-[4px] border border-neutral-200">
                  <table className="w-full border-collapse text-left text-sm">
                    <thead className="bg-neutral-100 text-xs uppercase tracking-wide text-neutral-600">
                      <tr>
                        <TableHead>Present State</TableHead>
                        <TableHead>{inputVariable || "X"}</TableHead>
                        <TableHead>Next State</TableHead>
                        <TableHead>{outputVariable || "Z"}</TableHead>
                      </tr>
                    </thead>
                    <tbody>
                      {stateRows.map((row, index) => (
                        <tr key={index} className="border-t border-neutral-200">
                          <EditableCell
                            value={row.presentState}
                            onChange={(value) =>
                              updateRow(index, "presentState", value)
                            }
                          />
                          <EditableCell
                            value={row.input}
                            onChange={(value) => updateRow(index, "input", value)}
                          />
                          <EditableCell
                            value={row.nextState}
                            onChange={(value) =>
                              updateRow(index, "nextState", value)
                            }
                          />
                          <EditableCell
                            value={row.output}
                            onChange={(value) => updateRow(index, "output", value)}
                          />
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <button
                  type="button"
                  onClick={handleGenerate}
                  className="mt-4 h-12 w-full rounded-[3px] border border-neutral-950 bg-neutral-950 px-5 text-sm font-semibold tracking-wide text-white transition hover:bg-neutral-800 focus:outline-none"
                >
                  GENERATE
                </button>
                <div className="mt-3 rounded-[4px] border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm font-medium text-neutral-700">
                  Status: {generationStatus}
                </div>
              </Panel>

              <Panel title="6. CSV Tools">
                <textarea
                  value={csvText}
                  onChange={(event) => setCsvText(event.target.value)}
                  placeholder="請以逗號分隔，填入順序請參考上方 state table"
                  className="min-h-20 w-full resize-y rounded-[3px] border border-neutral-300 bg-white px-3 py-2 text-sm leading-6 text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-neutral-950"
                />
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <SmallButton onClick={importCsv}>Import CSV</SmallButton>
                  <SmallButton onClick={exportCsv}>Export CSV</SmallButton>
                </div>
              </Panel>
          </aside>

          <div
            role="separator"
            aria-orientation="vertical"
            onPointerDown={startPaneResize}
            className="group flex w-4 shrink-0 cursor-col-resize items-stretch justify-center"
          >
            <div className="h-full w-px bg-neutral-200 transition group-hover:bg-neutral-950" />
          </div>

          <section className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-y-auto pl-4 pr-1">
            <Panel title="ROW 1: EQUATIONS SUMMARY">
              <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
                <div className="grid gap-4">
                  <ValidationBox analysis={outputAnalysis} />
                  <EquationSummary analysis={outputAnalysis} />
                </div>
                <button
                  type="button"
                  onClick={copyEquations}
                  disabled={!outputAnalysis}
                  className="h-10 self-start rounded-[3px] border border-neutral-300 bg-white px-5 text-sm font-semibold text-neutral-800 transition hover:border-neutral-950 hover:bg-neutral-950 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Copy Equations
                </button>
              </div>
            </Panel>

            <Panel title="ROW 2: K-MAP VISUALIZATIONS">
              <div className="min-w-0 resize overflow-auto rounded-[4px] border border-neutral-200 bg-white p-3">
                <KMap equation={selectedEquation} analysis={outputAnalysis} />
              </div>
            </Panel>

            <div className="min-h-[640px]">
              <Panel title="ROW 3: SEQUENTIAL CIRCUIT DIAGRAM" fill>
                <CircuitDiagram analysis={outputAnalysis} flipFlopType={flipFlopType} />
              </Panel>
            </div>

            <TimingDiagram stateRows={stateRows} triggerEdge={triggerEdge} />
          </section>
        </section>

        <footer className="border-t border-neutral-200 bg-white/80 px-5 py-4 backdrop-blur-xl">
          <div className="mx-auto flex max-w-[1920px] items-center justify-between gap-3">
            <button
              type="button"
              className="rounded-[3px] border border-neutral-300 bg-white px-4 py-3 text-sm font-semibold text-neutral-700 transition hover:border-neutral-950 hover:bg-neutral-950 hover:text-white"
            >
              Settings
            </button>
            <button
              type="button"
              onClick={handleGenerate}
              className="rounded-[3px] border border-neutral-950 bg-neutral-950 px-12 py-4 text-sm font-semibold tracking-wide text-white transition hover:bg-neutral-800 focus:outline-none"
            >
              GENERATE
            </button>
            <button
              type="button"
              onClick={exportReport}
              disabled={!outputAnalysis}
              className="rounded-[3px] border border-neutral-300 bg-white px-4 py-3 text-sm font-semibold text-neutral-700 transition hover:border-neutral-950 hover:bg-neutral-950 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              Export Report
            </button>
          </div>
        </footer>
      </div>
    </main>
  );
}

function analyzeCircuit(
  rows: StateRow[],
  flipFlopType: FlipFlopType,
  inputVariable: string,
  outputVariable: string,
): AnalysisResult {
  const cleanRows = rows
    .map((row) => ({
      presentState: row.presentState.trim().toUpperCase(),
      input: row.input.trim(),
      nextState: row.nextState.trim().toUpperCase(),
      output: row.output.trim(),
    }))
    .filter(
      (row) =>
        row.presentState || row.input || row.nextState || row.output,
    );
  const errors: string[] = [];
  if (cleanRows.length === 0) errors.push("State table is empty.");
  cleanRows.forEach((row, index) => {
    if (!row.presentState) errors.push(`Row ${index + 1}: missing present state.`);
    if (!["0", "1"].includes(row.input))
      errors.push(`Row ${index + 1}: input ${inputVariable} must be 0 or 1.`);
    if (!row.nextState) errors.push(`Row ${index + 1}: missing next state.`);
    if (!["0", "1"].includes(row.output))
      errors.push(`Row ${index + 1}: output ${outputVariable} must be 0 or 1.`);
  });

  const states = Array.from(
    new Set(
      cleanRows.flatMap((row) => [row.presentState, row.nextState]).filter(Boolean),
    ),
  ).sort();
  const bitCount = Math.max(1, Math.ceil(Math.log2(Math.max(1, states.length))));
  const stateBits = Array.from({ length: bitCount }, (_, index) => `Q${bitCount - index - 1}`);
  const stateAssignments = Object.fromEntries(
    states.map((state, index) => [state, index.toString(2).padStart(bitCount, "0")]),
  );
  const variableNames = [...stateBits, inputVariable];
  if (variableNames.length > 4) {
    errors.push("K-map display supports up to 8 states with one input variable.");
  }
  const totalTerms = 2 ** variableNames.length;
  const coveredTerms = cleanRows.map((row) =>
    bitsToIndex(`${stateAssignments[row.presentState] ?? "0".repeat(bitCount)}${row.input}`),
  );
  const baseDontCares = range(totalTerms).filter((term) => !coveredTerms.includes(term));
  const equations: EquationResult[] = [];

  stateBits.forEach((bitName, bitIndex) => {
    const dMinterms: number[] = [];
    const tMinterms: number[] = [];
    const jMinterms: number[] = [];
    const kMinterms: number[] = [];
    const jDontCares = [...baseDontCares];
    const kDontCares = [...baseDontCares];

    cleanRows.forEach((row) => {
      const presentBits = stateAssignments[row.presentState];
      const nextBits = stateAssignments[row.nextState];
      if (!presentBits || !nextBits) return;
      const term = bitsToIndex(`${presentBits}${row.input}`);
      const excitation = calculateExcitation(presentBits[bitIndex], nextBits[bitIndex], flipFlopType);
      if (excitation.D === "1") dMinterms.push(term);
      if (excitation.T === "1") tMinterms.push(term);
      if (excitation.J === "1") jMinterms.push(term);
      if (excitation.J === "X") jDontCares.push(term);
      if (excitation.K === "1") kMinterms.push(term);
      if (excitation.K === "X") kDontCares.push(term);
    });

    if (flipFlopType === "d") {
      equations.push(makeEquation(`D${bitName.slice(1)}`, dMinterms, baseDontCares, variableNames));
    }
    if (flipFlopType === "t") {
      equations.push(makeEquation(`T${bitName.slice(1)}`, tMinterms, baseDontCares, variableNames));
    }
    if (flipFlopType === "jk") {
      equations.push(makeEquation(`J${bitName.slice(1)}`, jMinterms, unique(jDontCares), variableNames));
      equations.push(makeEquation(`K${bitName.slice(1)}`, kMinterms, unique(kDontCares), variableNames));
    }
  });

  const outputMinterms = cleanRows
    .filter((row) => row.output === "1")
    .map((row) =>
      bitsToIndex(`${stateAssignments[row.presentState] ?? "0".repeat(bitCount)}${row.input}`),
    );
  const outputEquation = makeEquation(outputVariable, outputMinterms, baseDontCares, variableNames);

  return {
    valid: errors.length === 0,
    errors,
    states,
    bitCount,
    stateBits,
    stateAssignments,
    equations,
    outputEquation,
    variableNames,
  };
}

function makeEquation(
  signal: string,
  minterms: number[],
  dontCares: number[],
  variableNames: string[],
): EquationResult {
  const cleanMinterms = unique(minterms);
  const cleanDontCares = unique(dontCares).filter((term) => !cleanMinterms.includes(term));
  const simplified = simplifyBoolean(cleanMinterms, cleanDontCares, variableNames);
  const verified = verifyEquation(simplified.equation, cleanMinterms, cleanDontCares, variableNames);
  const fallback = canonicalSop(cleanMinterms, variableNames);
  return {
    signal,
    equation: verified ? simplified.equation : fallback.equation,
    minterms: cleanMinterms,
    dontCares: cleanDontCares,
    implicants: verified ? simplified.implicants : fallback.implicants,
    verified,
  };
}

function calculateExcitation(
  presentState: string,
  nextState: string,
  ffType: FlipFlopType,
) {
  if (ffType === "d") return { D: nextState, T: "X", J: "X", K: "X" };
  if (ffType === "t") return { D: "X", T: presentState === nextState ? "0" : "1", J: "X", K: "X" };
  if (presentState === "0" && nextState === "0") return { D: "X", T: "X", J: "0", K: "X" };
  if (presentState === "0" && nextState === "1") return { D: "X", T: "X", J: "1", K: "X" };
  if (presentState === "1" && nextState === "0") return { D: "X", T: "X", J: "X", K: "1" };
  return { D: "X", T: "X", J: "X", K: "0" };
}

function simplifyBoolean(
  minterms: number[],
  dontCares: number[],
  variableNames: string[],
) {
  const variableCount = variableNames.length;
  const totalTerms = 2 ** variableCount;
  if (minterms.length === 0) return { equation: "0", implicants: [] };
  if (minterms.length + dontCares.length === totalTerms) return { equation: "1", implicants: ["-".repeat(variableCount)] };
  const allTerms = unique([...minterms, ...dontCares]).map((term) => ({
    pattern: term.toString(2).padStart(variableCount, "0"),
    terms: [term],
  }));
  let current = allTerms;
  const primes: typeof allTerms = [];

  while (current.length) {
    const used = new Set<number>();
    const next: typeof allTerms = [];
    for (let i = 0; i < current.length; i += 1) {
      for (let j = i + 1; j < current.length; j += 1) {
        const combined = combinePatterns(current[i].pattern, current[j].pattern);
        if (combined) {
          used.add(i);
          used.add(j);
          const merged = unique([...current[i].terms, ...current[j].terms]);
          if (!next.some((term) => term.pattern === combined)) {
            next.push({ pattern: combined, terms: merged });
          }
        }
      }
    }
    current.forEach((term, index) => {
      if (!used.has(index) && !primes.some((prime) => prime.pattern === term.pattern)) {
        primes.push(term);
      }
    });
    current = next;
  }

  let selected: typeof primes = [];
  for (let size = 1; size <= primes.length; size += 1) {
    const covers = combinations(primes, size).filter((combo) =>
      minterms.every((term) =>
        combo.some((prime) => patternCovers(prime.pattern, term, variableCount)),
      ),
    );
    if (covers.length) {
      selected = covers.sort((a, b) => {
        const aLiteralCount = a.reduce((sum, item) => sum + countLiterals(item.pattern), 0);
        const bLiteralCount = b.reduce((sum, item) => sum + countLiterals(item.pattern), 0);
        return aLiteralCount - bLiteralCount;
      })[0];
      break;
    }
  }

  return {
    equation: selected.map((term) => patternToExpression(term.pattern, variableNames)).join(" + "),
    implicants: selected.map((term) => term.pattern),
  };
}

function verifyEquation(
  equationString: string,
  minterms: number[],
  dontCares: number[],
  variableNames: string[],
) {
  const mintermSet = new Set(minterms);
  const dontCareSet = new Set(dontCares);
  const totalTerms = 2 ** variableNames.length;
  for (let term = 0; term < totalTerms; term += 1) {
    if (dontCareSet.has(term)) continue;
    const values = termToValues(term, variableNames);
    const evaluated = evaluateEquation(equationString, values);
    if (mintermSet.has(term) && !evaluated) return false;
    if (!mintermSet.has(term) && evaluated) return false;
  }
  return true;
}

function evaluateEquation(equationString: string, values: Record<string, boolean>) {
  const normalized = equationString.trim();
  if (normalized === "1") return true;
  if (normalized === "0" || normalized === "") return false;
  return normalized.split("+").some((product) => {
    const literals = product
      .trim()
      .split(/\s*(?:\u00b7|\*|&|\s)\s*/u)
      .map((literal) => literal.trim())
      .filter(Boolean);
    return literals.every((literal) => {
      const inverted = literal.endsWith("'");
      const name = inverted ? literal.slice(0, -1) : literal;
      const value = values[name] ?? false;
      return inverted ? !value : value;
    });
  });
}

function termToValues(term: number, variableNames: string[]) {
  const bits = term.toString(2).padStart(variableNames.length, "0");
  return Object.fromEntries(variableNames.map((name, index) => [name, bits[index] === "1"]));
}

function canonicalSop(minterms: number[], variableNames: string[]) {
  if (minterms.length === 0) return { equation: "0", implicants: [] };
  return {
    equation: minterms
      .map((term) =>
        patternToExpression(term.toString(2).padStart(variableNames.length, "0"), variableNames),
      )
      .join(" + "),
    implicants: minterms.map((term) => term.toString(2).padStart(variableNames.length, "0")),
  };
}

function combinePatterns(a: string, b: string) {
  let differences = 0;
  let result = "";
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] === b[index]) result += a[index];
    else {
      differences += 1;
      result += "-";
    }
  }
  return differences === 1 ? result : null;
}

function patternCovers(pattern: string, term: number, variableCount: number) {
  const bits = term.toString(2).padStart(variableCount, "0");
  return [...pattern].every((bit, index) => bit === "-" || bit === bits[index]);
}

function patternToExpression(pattern: string, variableNames: string[]) {
  const parts = [...pattern].flatMap((bit, index) => {
    if (bit === "-") return [];
    return bit === "1" ? variableNames[index] : `${variableNames[index]}'`;
  });
  return parts.length ? parts.join(" \u00b7 ") : "1";
}

function combinations<T>(items: T[], size: number): T[][] {
  if (size === 0) return [[]];
  if (items.length < size) return [];
  const [first, ...rest] = items;
  return [
    ...combinations(rest, size - 1).map((combo) => [first, ...combo]),
    ...combinations(rest, size),
  ];
}

function countLiterals(pattern: string) {
  return [...pattern].filter((bit) => bit !== "-").length;
}

function bitsToIndex(bits: string) {
  return Number.parseInt(bits, 2);
}

function range(length: number) {
  return Array.from({ length }, (_, index) => index);
}

function unique(values: number[]) {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function Panel({
  title,
  children,
  fill = false,
}: {
  title: string;
  children: React.ReactNode;
  fill?: boolean;
}) {
  return (
    <section
      className={`rounded-[4px] border border-neutral-200 bg-white/90 p-4 shadow-none backdrop-blur ${
        fill ? "flex h-full min-h-0 flex-1 flex-col" : ""
      }`}
    >
      <h2 className="mb-4 border-b border-neutral-200 pb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-neutral-600">
        {title}
      </h2>
      <div className={fill ? "min-h-0 flex-1" : ""}>{children}</div>
    </section>
  );
}

function ChoiceCard({
  name,
  label,
  description,
  checked,
  onChange,
}: {
  name: string;
  label: string;
  description: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label
      className={`flex cursor-pointer gap-3 rounded-[4px] border p-3 transition ${
        checked
          ? "border-neutral-950 bg-neutral-50"
          : "border-neutral-200 bg-white hover:border-neutral-400 hover:bg-neutral-50"
      }`}
    >
      <input
        type="radio"
        name={name}
        checked={checked}
        onChange={onChange}
        className="mt-1 h-4 w-4 accent-neutral-950"
      />
      <span>
        <span className="block text-sm font-semibold text-neutral-950">{label}</span>
        <span className="mt-1 block text-xs leading-5 text-neutral-500">
          {description}
        </span>
      </span>
    </label>
  );
}

function SmallButton({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-[3px] border border-neutral-300 bg-white px-3 py-2 text-xs font-semibold text-neutral-700 transition hover:border-neutral-950 hover:bg-neutral-950 hover:text-white"
    >
      {children}
    </button>
  );
}

function EdgeTriggerControl({
  value,
  onChange,
}: {
  value: TriggerEdge;
  onChange: (value: TriggerEdge) => void;
}) {
  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-2 gap-3">
        <button
          type="button"
          onClick={() => onChange("rising")}
          className={`h-12 rounded-[3px] border px-4 text-sm font-semibold transition ${
            value === "rising"
              ? "border-neutral-950 bg-neutral-950 text-white"
              : "border-neutral-300 bg-white text-neutral-700 hover:border-neutral-950"
          }`}
        >
          Rising Edge 上緣
        </button>
        <button
          type="button"
          onClick={() => onChange("falling")}
          className={`h-12 rounded-[3px] border px-4 text-sm font-semibold transition ${
            value === "falling"
              ? "border-neutral-950 bg-neutral-950 text-white"
              : "border-neutral-300 bg-white text-neutral-700 hover:border-neutral-950"
          }`}
        >
          Falling Edge 下緣
        </button>
      </div>
      <div className="rounded-[4px] border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm font-medium text-neutral-700">
        Timing diagram trigger: {value === "rising" ? "Rising Edge" : "Falling Edge"}
      </div>
    </div>
  );
}

function TableHead({ children }: { children: React.ReactNode }) {
  return <th className="px-3 py-3 font-bold">{children}</th>;
}

function EditableCell({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <td className="p-2">
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-9 w-full rounded-[2px] border border-neutral-300 bg-white px-2 text-sm text-neutral-950 outline-none transition focus:border-neutral-950"
      />
    </td>
  );
}

function ValidationBox({ analysis }: { analysis: AnalysisResult | null }) {
  if (!analysis) {
    return (
      <div className="rounded-[4px] border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm text-neutral-600">
        Press Generate to validate the table and synthesize equations.
      </div>
    );
  }
  if (analysis.valid) {
    return (
      <div className="rounded-[4px] border border-emerald-300 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-950">
        Validation passed. Equations are generated from deterministic excitation rules.
      </div>
    );
  }
  return (
    <div className="rounded-[4px] border border-rose-300 bg-rose-50 px-4 py-3 text-sm text-rose-900">
      {analysis.errors.slice(0, 4).map((error) => (
        <div key={error}>{error}</div>
      ))}
    </div>
  );
}

function StateAssignmentBox({ analysis }: { analysis: AnalysisResult | null }) {
  if (!analysis) return null;
  return (
    <div className="rounded-[4px] border border-neutral-200 bg-white p-3">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-600">
        State Variables: {analysis.stateBits.join(" ")}
      </h3>
      <div className="grid grid-cols-3 gap-2 text-xs">
        {Object.entries(analysis.stateAssignments).map(([state, bits]) => (
          <div
            key={state}
            className="rounded-[3px] border border-neutral-200 bg-neutral-50 px-2 py-1 font-semibold text-neutral-700"
          >
            {state} = {bits}
          </div>
        ))}
      </div>
    </div>
  );
}

function CoverageMeter({
  rows,
  analysis,
}: {
  rows: StateRow[];
  analysis: AnalysisResult | null;
}) {
  const states = Array.from(
    new Set(
      rows
        .flatMap((row) => [row.presentState.trim(), row.nextState.trim()])
        .filter(Boolean),
    ),
  );
  const expected = Math.max(1, states.length * 2);
  const filled = new Set(
    rows
      .filter(
        (row) =>
          row.presentState.trim() &&
          ["0", "1"].includes(row.input.trim()) &&
          row.nextState.trim() &&
          ["0", "1"].includes(row.output.trim()),
      )
      .map((row) => `${row.presentState.trim().toUpperCase()}-${row.input.trim()}`),
  ).size;
  const percent = Math.min(100, Math.round((filled / expected) * 100));
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between text-xs font-black uppercase tracking-wide text-slate-600">
        <span>Table Coverage</span>
        <span>{analysis?.valid ? "Complete" : `${percent}%`}</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full ${
            analysis?.valid ? "bg-emerald-500" : "bg-cyan-500"
          }`}
          style={{ width: `${analysis?.valid ? 100 : percent}%` }}
        />
      </div>
    </div>
  );
}

function EquationTable({ analysis }: { analysis: AnalysisResult | null }) {
  const equations = analysis
    ? [...analysis.equations, analysis.outputEquation]
    : [];
  return (
    <div>
      <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-700">
        Flip-Flop Input Equations
      </h2>
      <div className="overflow-hidden rounded-lg border border-slate-200">
        <table className="w-full border-collapse text-left text-sm">
          <thead className="bg-slate-100 text-xs uppercase tracking-wide text-slate-600">
            <tr>
              <TableHead>Input</TableHead>
              <TableHead>Equation</TableHead>
            </tr>
          </thead>
          <tbody>
            {equations.length ? (
              equations.map((equation) => (
                <EquationRow
                  key={equation.signal}
                  label={equation.signal}
                  value={equation.equation}
                />
              ))
            ) : (
              <EquationRow label="-" value="Pending generation" />
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EquationSummary({ analysis }: { analysis: AnalysisResult | null }) {
  const equations = analysis
    ? [...analysis.equations, analysis.outputEquation]
    : [];
  if (!equations.length) {
    return (
      <div className="rounded-[4px] border border-neutral-200 bg-neutral-50 px-4 py-4 font-mono text-sm text-neutral-500">
        Pending generation
      </div>
    );
  }
  return (
    <div className="grid gap-2 rounded-[4px] border border-neutral-200 bg-white p-3">
      {equations.map((equation) => (
        <div
          key={equation.signal}
          className="flex items-center gap-3 border-b border-neutral-100 px-2 py-2 last:border-b-0"
        >
          <span className="w-12 font-mono text-sm font-bold text-neutral-950">
            {equation.signal}
          </span>
          <span className="text-neutral-400">=</span>
          <span className="font-mono text-sm font-semibold text-neutral-800">
            {equation.equation}
          </span>
        </div>
      ))}
    </div>
  );
}

function EquationTableOld({ analysis: _analysis }: { analysis: AnalysisResult | null }) {
  const equations = [
    { signal: "J1", equation: "Q0 繚 X" },
    { signal: "K1", equation: "Q0 繚 X" },
    { signal: "J0", equation: "X" },
    { signal: "K0", equation: "X" },
    { signal: "Z", equation: "Q1 繚 Q0 繚 X" },
  ];
  return (
    <div>
      <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-700">
        Flip-Flop Input Equations
      </h2>
      <div className="overflow-hidden rounded-lg border border-slate-200">
        <table className="w-full border-collapse text-left text-sm">
          <thead className="bg-slate-100 text-xs uppercase tracking-wide text-slate-600">
            <tr>
              <TableHead>Input</TableHead>
              <TableHead>Equation</TableHead>
            </tr>
          </thead>
          <tbody>
            {equations.map((equation) => (
              <EquationRow
                key={equation.signal}
                label={equation.signal}
                value={equation.equation}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EquationRow({ label, value }: { label: string; value: string }) {
  return (
    <tr className="border-t border-slate-100">
      <td className="px-3 py-3 font-black text-cyan-800">{label}</td>
      <td className="px-3 py-3 font-mono text-sm text-slate-700">{value}</td>
    </tr>
  );
}

function KMap({
  equation,
  analysis,
}: {
  equation: EquationResult | null;
  analysis: AnalysisResult | null;
}) {
  if (!analysis || !equation) {
    return (
      <div style={{ display: "grid", minHeight: "280px", placeItems: "center", border: "1px dashed #cbd5e1", borderRadius: "12px", backgroundColor: "#f8fafc", color: "#64748b", fontWeight: "bold" }}>
        Waiting for state table data
      </div>
    );
  }
  const equations = [...analysis.equations, analysis.outputEquation];
  if (analysis.variableNames.length !== 3) {
    return (
      <div style={{ padding: "20px", border: "1px solid #cbd5e1", borderRadius: "12px", backgroundColor: "#f8fafc", color: "#0f172a", fontWeight: "bold" }}>
        Dynamic minimization is active. K-Map rendering currently displays the standard 3-variable Q1, Q0, X topology.
      </div>
    );
  }
  return (
    <div style={{ display: "flex", flexWrap: "nowrap", gap: "16px", justifyContent: "flex-start", overflowX: "auto", paddingBottom: "8px" }}>
      {equations.map((item) => (
        <KMapInline
          key={item.signal}
          title={`K-Map: ${item.signal}`}
          equation={`${item.signal} = ${item.equation}`}
          data={buildKMapData(item, analysis.variableNames.length)}
        />
      ))}
    </div>
  );
}

function KMapStaticUnused({
  equation: _equation,
  analysis: _analysis,
}: {
  equation: EquationResult | null;
  analysis: AnalysisResult | null;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", justifyContent: "center" }}>
      <KMapInline title="K-Map: J1" equation="J1 = X 繚 Q0" data={[[{ val: "0" }, { val: "0" }, { val: "1", hl: true }, { val: "0" }], [{ val: "X" }, { val: "X" }, { val: "X", hl: true }, { val: "X" }]]} />
      <KMapInline title="K-Map: K1" equation="K1 = X 繚 Q0" data={[[{ val: "X" }, { val: "X" }, { val: "X", hl: true }, { val: "X" }], [{ val: "0" }, { val: "0" }, { val: "1", hl: true }, { val: "0" }]]} />
      <KMapInline title="K-Map: J0" equation="J0 = X" data={[[{ val: "0" }, { val: "1", hl: true }, { val: "X", hl: true }, { val: "X" }], [{ val: "0" }, { val: "1", hl: true }, { val: "X", hl: true }, { val: "X" }]]} />
      <KMapInline title="K-Map: K0" equation="K0 = X" data={[[{ val: "X" }, { val: "X" }, { val: "0" }, { val: "1", hl: true }], [{ val: "X" }, { val: "X" }, { val: "0" }, { val: "1", hl: true }]]} />
      <KMapInline title="K-Map: Z" equation="Z = Q1 繚 Q0 繚 X" data={[[{ val: "0" }, { val: "0" }, { val: "0" }, { val: "0" }], [{ val: "0" }, { val: "0" }, { val: "1", hl: true }, { val: "0" }]]} />
    </div>
  );
}

function KMapInline({
  title,
  equation,
  data,
}: {
  title: string;
  equation: string;
  data: Array<Array<{ val: string; hl?: boolean }>>;
}) {
  return (
    <div style={{ display: "inline-flex", flex: "0 0 auto", flexDirection: "column", alignItems: "center", margin: "0", padding: "20px", border: "1px solid #cbd5e1", borderRadius: "4px", backgroundColor: "#f8fafc", color: "#0f172a" }}>
      <h4 style={{ margin: "0 0 15px 0", fontSize: "18px", fontWeight: "bold" }}>{title}</h4>
      <table style={{ borderCollapse: "collapse", textAlign: "center", fontFamily: "monospace", fontSize: "18px" }}>
        <tbody>
          <tr>
            <td colSpan={2} rowSpan={2} style={{ border: "none" }} />
            <td colSpan={4} style={{ border: "none", paddingBottom: "8px", fontWeight: "bold" }}>Q0 X</td>
          </tr>
          <tr>
            <td style={{ border: "none", width: "50px", paddingBottom: "5px" }}>00</td>
            <td style={{ border: "none", width: "50px", paddingBottom: "5px" }}>01</td>
            <td style={{ border: "none", width: "50px", paddingBottom: "5px" }}>11</td>
            <td style={{ border: "none", width: "50px", paddingBottom: "5px" }}>10</td>
          </tr>
          <tr>
            <td rowSpan={2} style={{ border: "none", paddingRight: "8px", fontWeight: "bold", verticalAlign: "middle" }}>Q1</td>
            <td style={{ border: "none", paddingRight: "10px" }}>0</td>
            {data[0].map((cell, i) => (
              <td key={`r0-${i}`} style={{ border: "2px solid #334155", width: "50px", height: "50px", backgroundColor: cell.hl ? "#dcfce7" : "#ffffff", color: cell.hl ? "#166534" : "#0f172a", fontWeight: cell.hl ? "bold" : "normal", outline: cell.hl ? "3px solid #22c55e" : "none", outlineOffset: "-3px" }}>
                {cell.val}
              </td>
            ))}
          </tr>
          <tr>
            <td style={{ border: "none", paddingRight: "10px" }}>1</td>
            {data[1].map((cell, i) => (
              <td key={`r1-${i}`} style={{ border: "2px solid #334155", width: "50px", height: "50px", backgroundColor: cell.hl ? "#dcfce7" : "#ffffff", color: cell.hl ? "#166534" : "#0f172a", fontWeight: cell.hl ? "bold" : "normal", outline: cell.hl ? "3px solid #22c55e" : "none", outlineOffset: "-3px" }}>
                {cell.val}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      <div style={{ marginTop: "20px", color: "#1d4ed8", fontWeight: "bold", fontSize: "18px" }}>(Simplified) {equation}</div>
    </div>
  );
}

function buildKMapData(equation: EquationResult, variableCount: number) {
  const rows = ["0", "1"];
  const columns = ["00", "01", "11", "10"];
  return rows.map((row) =>
    columns.map((column) => {
      const term = bitsToIndex(getKMapBits(row, column, variableCount));
      const isMinterm = equation.minterms.includes(term);
      const isDontCare = equation.dontCares.includes(term);
      const isGrouped = equation.implicants.some((pattern) =>
        patternCovers(pattern, term, variableCount),
      );
      return {
        val: isMinterm ? "1" : isDontCare ? "X" : "0",
        hl: isGrouped && (isMinterm || isDontCare),
      };
    }),
  );
}

function KMapClassUnused({
  equation: _equation,
  analysis: _analysis,
}: {
  equation: EquationResult | null;
  analysis: AnalysisResult | null;
}) {
  return (
    <div className="grid grid-cols-1 gap-8 p-4 lg:grid-cols-2 xl:grid-cols-3">
      <KMapBox
        title="K-Map (Example) J1"
        equation="J1 = X 繚 Q0"
        row0={[{ val: "0" }, { val: "0" }, { val: "1", highlight: true }, { val: "0" }]}
        row1={[{ val: "X" }, { val: "X" }, { val: "X", highlight: true }, { val: "X" }]}
      />
      <KMapBox
        title="K-Map K1"
        equation="K1 = X 繚 Q0"
        row0={[{ val: "X" }, { val: "X" }, { val: "X", highlight: true }, { val: "X" }]}
        row1={[{ val: "0" }, { val: "0" }, { val: "1", highlight: true }, { val: "0" }]}
      />
      <KMapBox
        title="K-Map J0"
        equation="J0 = X"
        row0={[{ val: "0" }, { val: "1", highlight: true }, { val: "X", highlight: true }, { val: "X" }]}
        row1={[{ val: "0" }, { val: "1", highlight: true }, { val: "X", highlight: true }, { val: "X" }]}
      />
      <KMapBox
        title="K-Map K0"
        equation="K0 = X"
        row0={[{ val: "X" }, { val: "X" }, { val: "0" }, { val: "1", highlight: true }]}
        row1={[{ val: "X" }, { val: "X" }, { val: "0" }, { val: "1", highlight: true }]}
      />
      <KMapBox
        title="K-Map Z"
        equation="Z = Q1 繚 Q0 繚 X"
        row0={[{ val: "0" }, { val: "0" }, { val: "0" }, { val: "0" }]}
        row1={[{ val: "0" }, { val: "0" }, { val: "1", highlight: true }, { val: "0" }]}
      />
    </div>
  );
}

function KMapBox({
  title,
  equation,
  row0,
  row1,
}: {
  title: string;
  equation: string;
  row0: Array<{ val: string; highlight?: boolean }>;
  row1: Array<{ val: string; highlight?: boolean }>;
}) {
  return (
    <div className="flex w-full max-w-md flex-col items-center rounded-lg border border-gray-300 bg-white p-6 shadow-sm">
      <div className="mb-4 w-full text-left text-lg font-bold text-gray-800">{title}</div>
      <div className="relative mt-2 flex items-start">
        <div className="flex flex-col items-end pr-3 pt-7">
          <span className="mb-2 pr-1 text-sm font-bold">Q1</span>
          <span className="flex h-12 items-center font-medium text-gray-700">0</span>
          <span className="flex h-12 items-center font-medium text-gray-700">1</span>
        </div>
        <div>
          <div className="mb-1 flex flex-col items-center">
            <span className="mb-1 text-sm font-bold">Q0 X</span>
            <div className="flex w-full">
              <span className="w-12 text-center font-medium text-gray-700">00</span>
              <span className="w-12 text-center font-medium text-gray-700">01</span>
              <span className="w-12 text-center font-medium text-gray-700">11</span>
              <span className="w-12 text-center font-medium text-gray-700">10</span>
            </div>
          </div>
          <div className="grid grid-cols-4 border-l-2 border-t-2 border-gray-800 bg-white">
            {row0.map((cell, i) => (
              <div key={`r0-${i}`} className={`flex h-12 w-12 items-center justify-center border-b-2 border-r-2 border-gray-800 text-xl font-medium ${cell.highlight ? "z-10 scale-105 rounded-md border-4 border-green-500 bg-green-50 text-green-600 shadow-sm" : "text-gray-800"}`}>
                {cell.val}
              </div>
            ))}
            {row1.map((cell, i) => (
              <div key={`r1-${i}`} className={`flex h-12 w-12 items-center justify-center border-b-2 border-r-2 border-gray-800 text-xl font-medium ${cell.highlight ? "z-10 scale-105 rounded-md border-4 border-green-500 bg-green-50 text-green-600 shadow-sm" : "text-gray-800"}`}>
                {cell.val}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-6 font-mono text-lg font-bold text-blue-700">(Simplified) {equation}</div>
    </div>
  );
}

function KMapUnused({
  equation: _equation,
  analysis: _analysis,
}: {
  equation: EquationResult | null;
  analysis: AnalysisResult | null;
}) {
  const maps = [
    {
      name: "J1",
      rows: [
        [
          { value: "0" },
          { value: "0" },
          { value: "1", grouped: true },
          { value: "0" },
        ],
        [
          { value: "X" },
          { value: "X" },
          { value: "X", grouped: true },
          { value: "X" },
        ],
      ],
      equation: "J1 = Q0 繚 X",
    },
    {
      name: "K1",
      rows: [
        [
          { value: "X" },
          { value: "X" },
          { value: "X", grouped: true },
          { value: "X" },
        ],
        [
          { value: "0" },
          { value: "0" },
          { value: "1", grouped: true },
          { value: "0" },
        ],
      ],
      equation: "K1 = Q0 繚 X",
    },
    {
      name: "J0",
      rows: [
        [
          { value: "0" },
          { value: "1", grouped: true },
          { value: "X" },
          { value: "X" },
        ],
        [
          { value: "0" },
          { value: "1", grouped: true },
          { value: "X" },
          { value: "X" },
        ],
      ],
      equation: "J0 = X",
    },
    {
      name: "K0",
      rows: [
        [
          { value: "X" },
          { value: "X" },
          { value: "0" },
          { value: "1", grouped: true },
        ],
        [
          { value: "X" },
          { value: "X" },
          { value: "0" },
          { value: "1", grouped: true },
        ],
      ],
      equation: "K0 = X",
    },
    {
      name: "Z",
      rows: [
        [
          { value: "0" },
          { value: "0" },
          { value: "0" },
          { value: "0" },
        ],
        [
          { value: "0" },
          { value: "0" },
          { value: "1", grouped: true },
          { value: "0" },
        ],
      ],
      equation: "Z = Q1 繚 Q0 繚 X",
    },
  ];

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-5">
      <div className="grid grid-cols-1 gap-10 md:grid-cols-2">
        {maps.map((map) => (
          <KMapCard key={map.name} name={map.name} rows={map.rows} equation={map.equation} />
        ))}
      </div>
    </div>
  );
}

function KMapCard({
  name,
  rows,
  equation,
}: {
  name: string;
  rows: Array<Array<{ value: string; grouped?: boolean }>>;
  equation: string;
}) {
  return (
    <div className="flex flex-col items-start">
      <h4 className="mb-4 text-lg font-bold text-gray-200">K-MAP: {name}</h4>
      <div className="flex">
        <div className="mr-3 flex flex-col justify-end pb-2 font-medium text-gray-400">
          <div className="flex h-12 items-center justify-end">Q1 = 0</div>
          <div className="mt-1 flex h-12 items-center justify-end">Q1 = 1</div>
        </div>

        <div>
          <div className="mb-2 flex gap-1 text-center font-medium text-gray-400">
            <div className="w-12">00</div>
            <div className="w-12">01</div>
            <div className="w-12">11</div>
            <div className="w-12">10</div>
          </div>
          <div className="-mt-8 ml-16 text-sm text-gray-400">Q0 X</div>
          <div className="mb-1 flex gap-1">
            {rows[0].map((cell, index) => (
              <KMapCell key={`${name}-0-${index}`} value={cell.value} grouped={cell.grouped} />
            ))}
          </div>
          <div className="flex gap-1">
            {rows[1].map((cell, index) => (
              <KMapCell key={`${name}-1-${index}`} value={cell.value} grouped={cell.grouped} />
            ))}
          </div>
        </div>
      </div>
      <div className="mt-4 font-mono text-lg text-blue-400">(Simplified) {equation}</div>
    </div>
  );
}

function KMapCell({
  value,
  grouped,
}: {
  value: string;
  grouped?: boolean;
}) {
  return (
    <div
      className={`flex h-12 w-12 items-center justify-center border-2 text-xl font-bold ${
        grouped
          ? "border-green-500 bg-green-900/40 text-green-400"
          : "border-gray-600 text-gray-200"
      }`}
    >
      {value}
    </div>
  );
}

function KMapOld() {
  const columns = ["00", "01", "11", "10"];
  const rows = [
    { q1: "0", values: ["0", "0", "1", "0"] },
    { q1: "1", values: ["X", "X", "X", "X"] },
  ];
  return (
    <div className="rounded-lg border border-slate-300 bg-white p-4">
      <div className="mx-auto w-fit">
        <div className="ml-14 mb-2 text-center text-sm font-black text-slate-800">
          Q0 X
        </div>
        <div className="relative">
          <table className="border-collapse text-center text-sm">
            <thead>
              <tr>
                <th className="h-12 w-14 border-2 border-slate-500 bg-slate-50 align-middle font-black text-slate-800">
                  Q1
                </th>
                {columns.map((column) => (
                  <th
                    key={column}
                    className="h-12 w-12 border-2 border-slate-500 bg-slate-50 align-middle font-black text-slate-800"
                  >
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.q1}>
                  <th className="h-12 w-14 border-2 border-slate-500 bg-slate-50 align-middle font-black text-slate-800">
                    {row.q1}
                  </th>
                  {row.values.map((value, index) => (
                    <td
                      key={`${row.q1}-${columns[index]}`}
                      className={`h-12 w-12 border-2 border-slate-500 align-middle text-lg font-black ${
                        value === "1" || value === "X"
                          ? "bg-emerald-50 text-emerald-900"
                          : "bg-white text-slate-800"
                      }`}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div
            className="pointer-events-none absolute rounded-lg border-2 border-green-500"
            style={{
              left: "152px",
              top: "48px",
              width: "48px",
              height: "96px",
            }}
          />
        </div>
      </div>
      <div className="px-4 py-4 text-center text-sm font-black text-slate-800">
        (Simplified) J1 = Q0 繚 X
      </div>
    </div>
  );
}

function getKMapBits(row: string, col: string, variableCount: number) {
  if (variableCount === 1) return col[1];
  if (variableCount === 2) return col;
  if (variableCount === 3) return `${row}${col}`;
  return `${row}${col}`.slice(0, variableCount);
}

function CircuitDiagram({
  analysis,
  flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  const safeParse = (val: unknown) => {
    if (val === undefined || val === null || val === "") return "0";
    return String(val).replace(/·/g, "").replace(/~/g, "'").replace(/!/g, "'").replace(/\s+/g, "");
  };

  useEffect(() => {
    if (!svgRef.current || !analysis) return;
    setRenderError(null);

    try {
      const equations = Object.fromEntries(
        [...analysis.equations, analysis.outputEquation].map((equation) => [
          equation.signal,
          equation.equation,
        ]),
      ) as Record<string, string>;
      const flipFlopEquations = analysis.equations
        .map((equation) => {
          const match = equation.signal.match(/^([JKDTSR])(\d+)$/);
          if (!match) return null;
          return {
            name: `${match[1]}Q${match[2]}`,
            expression: safeParse(equations[equation.signal]),
          };
        })
        .filter((equation): equation is { name: string; expression: string } => equation !== null && equation.expression !== "0");
      const equationsList = [
        ...flipFlopEquations,
        {
          name: analysis.outputEquation.signal,
          expression: safeParse(equations[analysis.outputEquation.signal]),
          type: "output",
        },
      ].filter((equation) => equation.expression !== "0");
      const engineFlipFlopType = flipFlopType === "jk" ? "JK" : flipFlopType === "d" ? "D" : "T";

      const engineAnalysis = {
        variables: { state: analysis.stateBits },
        equations: equationsList,
        graph: {
          flipFlops: analysis.stateBits.map((name) => ({ name, type: engineFlipFlopType })),
        },
      };

      resetDiagramView();
      renderCircuitDiagram(svgRef.current, engineAnalysis);
      bindDiagramPan(svgRef.current);
    } catch (err) {
      console.error("Routing Engine Error:", err);
      setRenderError(err instanceof Error ? err.toString() : String(err));
    }
  }, [analysis, flipFlopType]);

  if (!analysis) {
    return (
      <div className="grid h-full min-h-[600px] place-items-center rounded-[4px] border border-dashed border-neutral-300 bg-white px-6 text-center">
        <div>
          <p className="text-base font-semibold text-neutral-800">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-neutral-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-[600px] w-full flex-col overflow-hidden rounded-[4px] border border-neutral-200 bg-white" style={{ userSelect: "none", WebkitUserSelect: "none" }}>
      <div className="z-10 flex items-center justify-between border-b border-neutral-200 bg-white/80 px-4 py-2 backdrop-blur">
        <span className="font-semibold text-neutral-700">Dynamic EDA Schematic (Pan & Zoom Enabled)</span>
        <span className="text-sm font-semibold text-neutral-950">學號: 1140518 | 姓名: 張立澄</span>
      </div>
      <div className="relative flex-1 cursor-grab overflow-hidden bg-[#fafafa] active:cursor-grabbing" style={{ userSelect: "none", WebkitUserSelect: "none" }}>
        {renderError && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-red-50 p-6">
            <div className="rounded-[4px] border border-red-200 bg-white p-4 text-red-600">
              <h3 className="mb-2 text-lg font-bold">Circuit Rendering Failed</h3>
              <p className="font-mono text-sm">{renderError}</p>
            </div>
          </div>
        )}
        <svg ref={svgRef} className="absolute inset-0 block h-full w-full" style={{ userSelect: "none", WebkitUserSelect: "none", touchAction: "none" }} />
      </div>
    </div>
  );
}

function CircuitDiagramGridRouterUnused({
  analysis,
  flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }

  type Point = { x: number; y: number };
  type Box = { x: number; y: number; w: number; h: number; id: string };
  type GatePlan = { signal: string; equation: string; target: Point; gate?: Box; inputs: Array<{ term: string; point: Point; highwayX: number }> };

  const inputName = analysis.variableNames.at(-1) ?? "X";
  const topRail: Record<string, number> = { [inputName]: 42, [`${inputName}'`]: 92, X: 42, "X'": 92 };
  const equationMap = Object.fromEntries(
    [...analysis.equations, analysis.outputEquation].map((equation) => [
      equation.signal,
      equation.equation.replaceAll("*", "·").replaceAll("&", "·").replaceAll("繚", "·").trim(),
    ]),
  ) as Record<string, string>;

  const ff1 = { x: 450, y: 190, w: 90, h: 130, id: "FF1" };
  const ff0 = { x: 690, y: 190, w: 90, h: 130, id: "FF0" };
  const keepOuts: Box[] = [ff1, ff0];
  const pins: Record<string, Point> =
    flipFlopType === "jk"
      ? {
          J1: { x: ff1.x, y: ff1.y + 36 },
          K1: { x: ff1.x, y: ff1.y + 100 },
          J0: { x: ff0.x, y: ff0.y + 36 },
          K0: { x: ff0.x, y: ff0.y + 100 },
        }
      : {
          [`${flipFlopType.toUpperCase()}1`]: { x: ff1.x, y: ff1.y + 64 },
          [`${flipFlopType.toUpperCase()}0`]: { x: ff0.x, y: ff0.y + 64 },
        };

  const qPins: Record<string, Point> = {
    Q1: { x: ff1.x + ff1.w, y: ff1.y + 32 },
    "Q1'": { x: ff1.x + ff1.w, y: ff1.y + 100 },
    Q0: { x: ff0.x + ff0.w, y: ff0.y + 32 },
    "Q0'": { x: ff0.x + ff0.w, y: ff0.y + 100 },
  };

  const expandedBox = (box: Box, pad = 8) => ({
    x: box.x - pad,
    y: box.y - pad,
    w: box.w + pad * 2,
    h: box.h + pad * 2,
    id: box.id,
  });

  const segmentIntersectsBox = (a: Point, b: Point, box: Box) => {
    const k = expandedBox(box);
    if (a.x === b.x) {
      const minY = Math.min(a.y, b.y);
      const maxY = Math.max(a.y, b.y);
      return a.x > k.x && a.x < k.x + k.w && maxY > k.y && minY < k.y + k.h;
    }
    if (a.y === b.y) {
      const minX = Math.min(a.x, b.x);
      const maxX = Math.max(a.x, b.x);
      return a.y > k.y && a.y < k.y + k.h && maxX > k.x && minX < k.x + k.w;
    }
    return true;
  };

  const pathClear = (points: Point[], boxes: Box[]) =>
    points.slice(0, -1).every((point, index) =>
      boxes.every((box) => !segmentIntersectsBox(point, points[index + 1], box)),
    );

  const routeWire = (start: Point, end: Point, boxes: Box[], preferred?: { x?: number; y?: number }) => {
    const xs = [
      preferred?.x,
      start.x,
      end.x,
      Math.min(start.x, end.x) - 40,
      Math.min(start.x, end.x) - 80,
      Math.max(start.x, end.x) + 40,
      Math.max(start.x, end.x) + 80,
      250,
      320,
      610,
      840,
    ].filter((value): value is number => Number.isFinite(value));
    const ys = [
      preferred?.y,
      start.y,
      end.y,
      120,
      150,
      360,
      420,
      480,
      500,
      520,
      540,
      570,
    ].filter((value): value is number => Number.isFinite(value));

    const candidates: Point[][] = [];
    xs.forEach((x) => candidates.push([start, { x, y: start.y }, { x, y: end.y }, end]));
    ys.forEach((y) => candidates.push([start, { x: start.x, y }, { x: end.x, y }, end]));
    xs.forEach((x) =>
      ys.forEach((y) =>
        candidates.push([start, { x, y: start.y }, { x, y }, { x: end.x, y }, end]),
      ),
    );

    return candidates.find((candidate) => pathClear(candidate, boxes)) ?? candidates[candidates.length - 1];
  };

  const parseTerms = (equation: string) => {
    const clean = equation.replace(/\s+/g, "").replaceAll("繚", "·");
    if (!clean || clean === "0" || clean === "1") return { kind: clean || "0", terms: [] };
    if (clean.includes("+")) return { kind: "or", terms: clean.split("+") };
    if (clean.includes("·")) return { kind: "and", terms: clean.split("·") };
    return { kind: "literal", terms: [clean] };
  };

  const horizontalTrack = (() => {
    let next = 480;
    const assigned: Record<string, number> = {};
    return (signal: string) => {
      if (!assigned[signal]) {
        assigned[signal] = next;
        next += 20;
      }
      return assigned[signal];
    };
  })();

  const verticalTrack = (() => {
    const used = new Set<number>();
    return (base: number) => {
      let x = base;
      while (used.has(x)) x -= 20;
      used.add(x);
      return x;
    };
  })();

  const calculatePinYOffset = (numInputs: number, index: number) => {
    if (numInputs === 1) return 0;
    if (numInputs === 2) return index === 0 ? -12 : 12;
    return [-15, 0, 15][index] ?? 0;
  };

  const gatePlans: GatePlan[] = Object.entries(pins)
    .map(([signal, target]) => ({ signal, equation: equationMap[signal] ?? "0", target }))
    .concat([{ signal: analysis.outputEquation.signal, equation: equationMap[analysis.outputEquation.signal] ?? "0", target: { x: 860, y: 420 } }])
    .map((plan, index) => {
      const parsed = parseTerms(plan.equation);
      if (parsed.kind === "and" || parsed.kind === "or") {
        const gate = { x: plan.target.x - 76, y: plan.target.y - 24, w: 48, h: 48, id: `gate-${plan.signal}` };
        keepOuts.push(gate);
        return {
          ...plan,
          gate,
          inputs: parsed.terms.map((term, termIndex) => ({
            term,
            point: { x: gate.x + 5, y: plan.target.y + calculatePinYOffset(parsed.terms.length, termIndex) },
            highwayX: verticalTrack(gate.x - 20 - termIndex * 20),
          })),
        };
      }
      return {
        ...plan,
        inputs: parsed.terms.map((term) => ({
          term,
          point: plan.target,
          highwayX: verticalTrack(plan.target.x - 42 - index * 6),
        })),
      };
    });

  const feedbackRoutes = Object.entries(qPins).map(([signal, point]) => {
    const trackY = horizontalTrack(signal);
    const leftPoint = { x: 260, y: trackY };
    return {
      signal,
      trackY,
      points: routeWire(point, leftPoint, keepOuts, { y: trackY }),
    };
  });

  const sourcePoint = (term: string, highwayX: number): Point | null => {
    if (topRail[term]) return { x: highwayX, y: topRail[term] };
    const feedback = feedbackRoutes.find((route) => route.signal === term);
    if (feedback) return { x: highwayX, y: feedback.trackY };
    return null;
  };

  const renderPath = (points: Point[], key: string, stroke = "#0f172a", markerEnd?: string) => (
    <polyline
      key={key}
      points={points.map((point) => `${point.x},${point.y}`).join(" ")}
      fill="none"
      stroke={stroke}
      strokeWidth="2"
      markerEnd={markerEnd}
    />
  );

  const renderJunction = (point: Point, key: string) => <circle key={key} cx={point.x} cy={point.y} r="4" fill="#0f172a" />;

  return (
    <div className="h-full min-h-[680px] overflow-x-auto rounded-lg border border-gray-300 bg-white p-4 shadow-sm">
      <svg viewBox="0 0 960 680" className="h-full w-full min-w-[960px]">
        <defs>
          <marker id="grid-router-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a" />
          </marker>
        </defs>
        <rect x="0" y="0" width="960" height="680" fill="#ffffff" />
        <text x="24" y="28" fontSize="18" fontWeight="bold" fill="#0f172a">Grid-Based Channel Routed Schematic</text>
        <line x1="50" y1={topRail[inputName]} x2="920" y2={topRail[inputName]} stroke="#0f172a" strokeWidth="2" />
        <text x="20" y={topRail[inputName] + 5} fontSize="16" fontWeight="bold" fill="#0f172a">{inputName}</text>
        <circle cx="82" cy={topRail[inputName]} r="4" fill="#0f172a" />
        <line x1="82" y1={topRail[inputName]} x2="82" y2="56" stroke="#0f172a" strokeWidth="2" />
        <path d="M 72 56 L 92 56 L 82 72 Z" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="82" cy="75" r="3" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="82" y1="78" x2="82" y2={topRail[`${inputName}'`]} stroke="#0f172a" strokeWidth="2" />
        <line x1="50" y1={topRail[`${inputName}'`]} x2="920" y2={topRail[`${inputName}'`]} stroke="#0f172a" strokeWidth="2" />
        <text x="20" y={topRail[`${inputName}'`] + 5} fontSize="16" fontWeight="bold" fill="#0f172a">{inputName}'</text>
        <line x1="50" y1="630" x2="920" y2="630" stroke="#0f172a" strokeWidth="2" />
        <text x="10" y="635" fontSize="16" fontWeight="bold" fill="#0f172a">CLK</text>

        {feedbackRoutes.map((route) => renderPath(route.points, `fb-${route.signal}`))}
        {feedbackRoutes.map((route) => renderJunction(qPins[route.signal], `fb-dot-${route.signal}`))}

        {gatePlans.flatMap((plan) =>
          plan.inputs.flatMap((input, inputIndex) => {
            const source = sourcePoint(input.term, input.highwayX);
            if (!source) return [];
            const path = [
              { x: input.highwayX, y: source.y },
              { x: input.highwayX, y: input.point.y },
              input.point,
            ];
            return [renderPath(path, `${plan.signal}-${input.term}-${inputIndex}`), renderJunction(source, `${plan.signal}-${input.term}-${inputIndex}-dot`)];
          }),
        )}

        {gatePlans.map((plan) => {
          if (!plan.gate) return null;
          const parsed = parseTerms(plan.equation);
          return parsed.kind === "or" ? (
            <EdaOrGate key={`gate-${plan.signal}`} x={plan.gate.x} y={plan.gate.y} width={plan.gate.w} height={plan.gate.h} />
          ) : (
            <EdaAndGate key={`gate-${plan.signal}`} x={plan.gate.x} y={plan.gate.y} width={plan.gate.w} height={plan.gate.h} />
          );
        })}

        {gatePlans.map((plan) => {
          const parsed = parseTerms(plan.equation);
          if (parsed.kind === "1" || parsed.kind === "0") {
            return (
              <g key={`const-${plan.signal}`}>
                <text x={plan.target.x - 34} y={plan.target.y + 5} fontSize="14" fontWeight="bold" fill="#0f172a">{parsed.kind}</text>
                {renderPath([{ x: plan.target.x - 20, y: plan.target.y }, plan.target], `const-wire-${plan.signal}`, "#0f172a", plan.signal === analysis.outputEquation.signal ? "url(#grid-router-arrow)" : undefined)}
              </g>
            );
          }
          if (!plan.gate) return null;
          const output = { x: plan.gate.x + plan.gate.w, y: plan.target.y };
          return renderPath([output, plan.target], `out-${plan.signal}`, "#0f172a", plan.signal === analysis.outputEquation.signal ? "url(#grid-router-arrow)" : undefined);
        })}

        <TraditionalFlipFlop x={ff1.x} y={ff1.y} bit="1" type={flipFlopType} />
        <TraditionalFlipFlop x={ff0.x} y={ff0.y} bit="0" type={flipFlopType} />
        <line x1={ff1.x + 45} y1={ff1.y + ff1.h} x2={ff1.x + 45} y2="630" stroke="#0f172a" strokeWidth="2" />
        <line x1={ff0.x + 45} y1={ff0.y + ff0.h} x2={ff0.x + 45} y2="630" stroke="#0f172a" strokeWidth="2" />
        <circle cx={ff1.x + 45} cy="630" r="4" fill="#0f172a" />
        <circle cx={ff0.x + 45} cy="630" r="4" fill="#0f172a" />
        <text x="872" y="426" fontSize="16" fontWeight="bold" fill="#0f172a">{analysis.outputEquation.signal}</text>
      </svg>
    </div>
  );
}

function CircuitDiagramHardcodedUnused({
  analysis,
  flipFlopType: _flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }

  const equations = Object.fromEntries(
    [...analysis.equations, analysis.outputEquation].map((equation) => [
      equation.signal,
      equation.equation.replaceAll("*", "·").replaceAll("&", "·").replaceAll("繚", "·").trim(),
    ]),
  ) as Record<string, string>;
  const inputName = analysis.variableNames.at(-1) ?? "X";
  const TOP_RAIL: Record<string, number> = { [inputName]: 40, [`${inputName}'`]: 90, X: 40, "X'": 90 };
  const FB_Y: Record<string, number> = { Q1: 480, "Q1'": 500, Q0: 520, "Q0'": 540 };

  const drawWire = (term: string, targetX: number, targetY: number, highwayX: number) => {
    if (TOP_RAIL[term]) {
      return (
        <g>
          <line x1={highwayX} y1={TOP_RAIL[term]} x2={highwayX} y2={targetY} stroke="#0f172a" strokeWidth="2" />
          <circle cx={highwayX} cy={TOP_RAIL[term]} r="4" fill="#0f172a" />
          <line x1={highwayX} y1={targetY} x2={targetX} y2={targetY} stroke="#0f172a" strokeWidth="2" />
        </g>
      );
    }
    if (FB_Y[term]) {
      return (
        <g>
          <line x1={highwayX} y1={FB_Y[term]} x2={highwayX} y2={targetY} stroke="#0f172a" strokeWidth="2" />
          <circle cx={highwayX} cy={FB_Y[term]} r="4" fill="#0f172a" />
          <line x1={highwayX} y1={targetY} x2={targetX} y2={targetY} stroke="#0f172a" strokeWidth="2" />
        </g>
      );
    }
    return null;
  };

  const renderGate = (equation: string | undefined, targetPinX: number, targetPinY: number, isZ = false) => {
    if (!equation) return null;
    const eq = equation.replace(/\s+/g, "").replaceAll("繚", "·");
    const marker = isZ ? "url(#direct-arrow)" : undefined;

    if (eq === "1" || eq === "VCC") {
      return (
        <g>
          <text x={targetPinX - 30} y={targetPinY + 5} fontSize="14" fontWeight="bold" fill="#0f172a">1</text>
          <line x1={targetPinX - 20} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#0f172a" strokeWidth="2" markerEnd={marker} />
        </g>
      );
    }

    if (eq === "0" || eq === "GND") {
      return (
        <g>
          <text x={targetPinX - 30} y={targetPinY + 5} fontSize="14" fontWeight="bold" fill="#0f172a">0</text>
          <line x1={targetPinX - 20} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#0f172a" strokeWidth="2" markerEnd={marker} />
        </g>
      );
    }

    const gateX = targetPinX - 50;

    if (eq.includes("+")) {
      const terms = eq.split("+");
      return (
        <g>
          <path d={`M ${gateX} ${targetPinY - 20} Q ${gateX + 15} ${targetPinY} ${gateX} ${targetPinY + 20} Q ${gateX + 30} ${targetPinY + 20} ${gateX + 40} ${targetPinY} Q ${gateX + 30} ${targetPinY - 20} ${gateX} ${targetPinY - 20} Z`} fill="none" stroke="#0f172a" strokeWidth="2" />
          <line x1={gateX + 40} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#0f172a" strokeWidth="2" markerEnd={marker} />
          {drawWire(terms[0], gateX + 5, targetPinY - 10, gateX - 20)}
          {drawWire(terms[1], gateX + 5, targetPinY + 10, gateX - 40)}
        </g>
      );
    }

    if (eq.includes("·")) {
      const terms = eq.split("·");
      return (
        <g>
          <path d={`M ${gateX} ${targetPinY - 20} L ${gateX + 20} ${targetPinY - 20} A 20 20 0 0 1 ${gateX + 20} ${targetPinY + 20} L ${gateX} ${targetPinY + 20} Z`} fill="none" stroke="#0f172a" strokeWidth="2" />
          <line x1={gateX + 40} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#0f172a" strokeWidth="2" markerEnd={marker} />
          {terms.length === 2 ? (
            <>
              {drawWire(terms[0], gateX, targetPinY - 10, gateX - 20)}
              {drawWire(terms[1], gateX, targetPinY + 10, gateX - 40)}
            </>
          ) : null}
          {terms.length === 3 ? (
            <>
              {drawWire(terms[0], gateX, targetPinY - 15, gateX - 20)}
              {drawWire(terms[1], gateX, targetPinY, gateX - 40)}
              {drawWire(terms[2], gateX, targetPinY + 15, gateX - 60)}
            </>
          ) : null}
        </g>
      );
    }

    return (
      <g>
        {drawWire(eq, targetPinX, targetPinY, targetPinX - 40)}
      </g>
    );
  };

  return (
    <div className="h-full min-h-[650px] overflow-x-auto rounded-lg border border-gray-300 bg-white p-4 shadow-sm">
      <svg viewBox="0 0 900 650" className="h-full w-full min-w-[900px]">
        <defs>
          <marker id="direct-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a" />
          </marker>
        </defs>

        <rect x="0" y="0" width="900" height="650" fill="#ffffff" />
        <text x="24" y="28" fontSize="18" fontWeight="bold" fill="#0f172a">Direct Point-to-Point EDA Routing</text>

        <line x1="50" y1="40" x2="850" y2="40" stroke="#0f172a" strokeWidth="2" />
        <text x="20" y="45" fontSize="16" fontWeight="bold" fill="#0f172a">{inputName}</text>
        <circle cx="80" cy="40" r="4" fill="#0f172a" />
        <line x1="80" y1="40" x2="80" y2="55" stroke="#0f172a" strokeWidth="2" />
        <path d="M 70 55 L 90 55 L 80 70 Z" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="80" cy="73" r="3" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="80" y1="76" x2="80" y2="90" stroke="#0f172a" strokeWidth="2" />
        <line x1="50" y1="90" x2="850" y2="90" stroke="#0f172a" strokeWidth="2" />
        <text x="20" y="95" fontSize="16" fontWeight="bold" fill="#0f172a">{inputName}'</text>

        <line x1="50" y1="600" x2="850" y2="600" stroke="#0f172a" strokeWidth="2" />
        <text x="10" y="605" fontSize="16" fontWeight="bold" fill="#0f172a">CLK</text>

        <rect x="350" y="200" width="80" height="120" fill="none" stroke="#0f172a" strokeWidth="2" />
        <text x="360" y="230" fontSize="16" fill="#0f172a">J1</text>
        <text x="360" y="310" fontSize="16" fill="#0f172a">K1</text>
        <text x="400" y="230" fontSize="16" fill="#0f172a">Q1</text>
        <text x="395" y="310" fontSize="16" fill="#0f172a">Q1'</text>
        <polyline points="380,320 390,310 400,320" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="390" y1="320" x2="390" y2="600" stroke="#0f172a" strokeWidth="2" />
        <circle cx="390" cy="600" r="4" fill="#0f172a" />
        <polyline points="430,230 445,230 445,480 250,480" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="430" cy="230" r="4" fill="#0f172a" />
        <polyline points="430,310 455,310 455,500 250,500" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="430" cy="310" r="4" fill="#0f172a" />

        <rect x="600" y="200" width="80" height="120" fill="none" stroke="#0f172a" strokeWidth="2" />
        <text x="610" y="230" fontSize="16" fill="#0f172a">J0</text>
        <text x="610" y="310" fontSize="16" fill="#0f172a">K0</text>
        <text x="650" y="230" fontSize="16" fill="#0f172a">Q0</text>
        <text x="645" y="310" fontSize="16" fill="#0f172a">Q0'</text>
        <polyline points="630,320 640,310 650,320" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="640" y1="320" x2="640" y2="600" stroke="#0f172a" strokeWidth="2" />
        <circle cx="640" cy="600" r="4" fill="#0f172a" />
        <polyline points="680,230 695,230 695,520 250,520" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="680" cy="230" r="4" fill="#0f172a" />
        <polyline points="680,310 705,310 705,540 250,540" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="680" cy="310" r="4" fill="#0f172a" />

        {renderGate(equations.J1 ?? equations.D1 ?? equations.T1, 350, 225)}
        {renderGate(equations.K1, 350, 305)}
        {renderGate(equations.J0 ?? equations.D0 ?? equations.T0, 600, 225)}
        {renderGate(equations.K0, 600, 305)}
        {renderGate(equations[analysis.outputEquation.signal], 780, 420, true)}
        {equations[analysis.outputEquation.signal] ? <text x="790" y="425" fontSize="16" fontWeight="bold" fill="#0f172a">{analysis.outputEquation.signal}</text> : null}
      </svg>
    </div>
  );
}

function CircuitDiagramPreviousUnused({
  analysis,
  flipFlopType: _flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }
  const RAIL_X: Record<string, number> = { X: 50, "X'": 80, Q1: 110, "Q1'": 140, Q0: 170, "Q0'": 200 };
  const equationMap = Object.fromEntries(
    [...analysis.equations, analysis.outputEquation].map((equation) => [
      equation.signal,
      equation.equation.replaceAll("繚", "·").trim(),
    ]),
  );
  const dynamicJ1Equation = equationMap.J1 ?? equationMap.D1 ?? equationMap.T1 ?? "0";
  const dynamicK1Equation = equationMap.K1 ?? "0";
  const dynamicJ0Equation = equationMap.J0 ?? equationMap.D0 ?? equationMap.T0 ?? "0";
  const dynamicK0Equation = equationMap.K0 ?? "0";
  const zEquation = equationMap[analysis.outputEquation.signal] ?? "0";

  const renderInputLogic = (
    equation: string,
    targetPinX: number,
    targetPinY: number,
    pinName: string,
  ) => {
    const normalized = equation.replaceAll("*", "繚").replaceAll("&", "繚").trim();
    if (normalized === "1" || normalized === "VCC") {
      return (
        <g key={pinName}>
          <text x={targetPinX - 30} y={targetPinY + 5} fontSize="14" fontWeight="bold" fill="#0f172a">1</text>
          <line x1={targetPinX - 20} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#0f172a" strokeWidth="2" />
        </g>
      );
    }

    if (normalized === "0" || normalized === "GND") {
      return (
        <g key={pinName}>
          <text x={targetPinX - 30} y={targetPinY + 5} fontSize="14" fontWeight="bold" fill="#0f172a">0</text>
          <line x1={targetPinX - 20} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#0f172a" strokeWidth="2" />
        </g>
      );
    }

    if (RAIL_X[normalized] !== undefined) {
      const railX = RAIL_X[normalized];
      return (
        <g key={pinName}>
          <line x1={railX} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#0f172a" strokeWidth="2" />
          <circle cx={railX} cy={targetPinY} r="4" fill="#0f172a" />
        </g>
      );
    }

    if (normalized.includes("繚")) {
      const terms = normalized.split("繚").map((term) => term.trim());
      const gateX = targetPinX - 80;
      const gateY = targetPinY;
      const rail1 = RAIL_X[terms[0]] || 50;
      const rail2 = RAIL_X[terms[1]] || 50;

      return (
        <g key={pinName}>
          <path d={`M ${gateX} ${gateY - 20} L ${gateX + 15} ${gateY - 20} A 20 20 0 0 1 ${gateX + 15} ${gateY + 20} L ${gateX} ${gateY + 20} Z`} fill="none" stroke="#0f172a" strokeWidth="2" />
          <line x1={gateX + 35} y1={gateY} x2={targetPinX} y2={gateY} stroke="#0f172a" strokeWidth="2" />
          <line x1={rail1} y1={gateY - 10} x2={gateX} y2={gateY - 10} stroke="#0f172a" strokeWidth="2" />
          <circle cx={rail1} cy={gateY - 10} r="4" fill="#0f172a" />
          <line x1={rail2} y1={gateY + 10} x2={gateX} y2={gateY + 10} stroke="#0f172a" strokeWidth="2" />
          <circle cx={rail2} cy={gateY + 10} r="4" fill="#0f172a" />
        </g>
      );
    }

    return (
      <g key={pinName}>
        <text x={targetPinX - 60} y={targetPinY - 5} fontSize="12" fill="#d97706">{normalized}</text>
        <line x1={targetPinX - 40} y1={targetPinY} x2={targetPinX} y2={targetPinY} stroke="#d97706" strokeWidth="2" strokeDasharray="4" />
      </g>
    );
  };

  return (
    <div className="h-full min-h-[620px] overflow-x-auto rounded-lg border border-gray-300 bg-white p-4 shadow-sm">
      <svg viewBox="0 0 820 600" className="h-full w-full min-w-[820px]">
        <defs>
          <marker id="deterministic-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a" />
          </marker>
        </defs>

        <rect x="0" y="0" width="820" height="600" fill="#ffffff" />
        <text x="24" y="30" fill="#0f172a" fontSize="18" fontWeight="800">Deterministic EDA Schematic</text>

        {Object.entries(RAIL_X).map(([name, x]) => (
          <g key={name}>
            <line x1={x} y1="80" x2={x} y2="540" stroke="#0f172a" strokeWidth="2" />
            <text x={x} y="64" textAnchor="middle" fontSize="13" fontWeight="bold" fill="#0f172a">{name}</text>
          </g>
        ))}

        <line x1="20" y1="100" x2="50" y2="100" stroke="#0f172a" strokeWidth="2" />
        <text x="10" y="105" fontSize="14" fontWeight="bold" fill="#0f172a">X</text>
        <circle cx="50" cy="100" r="4" fill="#0f172a" />
        <line x1="50" y1="120" x2="58" y2="120" stroke="#0f172a" strokeWidth="2" />
        <circle cx="50" cy="120" r="4" fill="#0f172a" />
        <path d="M 58 108 L 58 132 L 78 120 Z" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
        <circle cx="84" cy="120" r="4" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
        <line x1="88" y1="120" x2="80" y2="120" stroke="#0f172a" strokeWidth="2" />
        <circle cx="80" cy="120" r="4" fill="#0f172a" />

        <rect x="350" y="190" width="80" height="140" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
        <text x="360" y="235" fontSize="16" fontWeight="bold" fill="#0f172a">J1</text>
        <text x="360" y="315" fontSize="16" fontWeight="bold" fill="#0f172a">K1</text>
        <text x="395" y="235" fontSize="16" fontWeight="bold" fill="#0f172a">Q1</text>
        <text x="390" y="315" fontSize="16" fontWeight="bold" fill="#0f172a">Q1'</text>
        <text x="390" y="268" textAnchor="middle" fontSize="16" fontWeight="bold" fill="#0f172a">JK</text>
        <polyline points="380,330 390,318 400,330" fill="none" stroke="#0f172a" strokeWidth="2" />

        <rect x="600" y="190" width="80" height="140" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
        <text x="610" y="235" fontSize="16" fontWeight="bold" fill="#0f172a">J0</text>
        <text x="610" y="315" fontSize="16" fontWeight="bold" fill="#0f172a">K0</text>
        <text x="645" y="235" fontSize="16" fontWeight="bold" fill="#0f172a">Q0</text>
        <text x="640" y="315" fontSize="16" fontWeight="bold" fill="#0f172a">Q0'</text>
        <text x="640" y="268" textAnchor="middle" fontSize="16" fontWeight="bold" fill="#0f172a">JK</text>
        <polyline points="630,330 640,318 650,330" fill="none" stroke="#0f172a" strokeWidth="2" />

        <line x1="20" y1="560" x2="760" y2="560" stroke="#0f172a" strokeWidth="2" />
        <text x="8" y="565" fontSize="14" fontWeight="bold" fill="#0f172a">CLK</text>
        <line x1="390" y1="330" x2="390" y2="560" stroke="#0f172a" strokeWidth="2" />
        <line x1="640" y1="330" x2="640" y2="560" stroke="#0f172a" strokeWidth="2" />
        <circle cx="390" cy="560" r="4" fill="#0f172a" />
        <circle cx="640" cy="560" r="4" fill="#0f172a" />

        <polyline points="430,230 430,450 110,450 110,80" fill="none" stroke="#0f172a" strokeWidth="2" />
        <polyline points="430,310 430,470 140,470 140,80" fill="none" stroke="#0f172a" strokeWidth="2" />
        <polyline points="680,230 680,490 170,490 170,80" fill="none" stroke="#0f172a" strokeWidth="2" />
        <polyline points="680,310 680,510 200,510 200,80" fill="none" stroke="#0f172a" strokeWidth="2" />

        <line x1="430" y1="230" x2="760" y2="230" stroke="#0f172a" strokeWidth="2" markerEnd="url(#deterministic-arrow)" />
        <text x="770" y="235" fontSize="16" fontWeight="bold" fill="#0f172a">Q1</text>
        <polyline points="680,230 710,230 710,250 760,250" fill="none" stroke="#0f172a" strokeWidth="2" markerEnd="url(#deterministic-arrow)" />
        <text x="770" y="255" fontSize="16" fontWeight="bold" fill="#0f172a">Q0</text>

        {renderInputLogic(dynamicJ1Equation, 350, 230, "J1")}
        {renderInputLogic(dynamicK1Equation, 350, 310, "K1")}
        {renderInputLogic(dynamicJ0Equation, 600, 230, "J0")}
        {renderInputLogic(dynamicK0Equation, 600, 310, "K0")}

        <text x="514" y="408" fontSize="12" fill="#d97706">{zEquation}</text>
        <line x1="520" y1="420" x2="760" y2="420" stroke="#d97706" strokeWidth="2" strokeDasharray="4" markerEnd="url(#deterministic-arrow)" />
        <text x="770" y="425" fontSize="16" fontWeight="bold" fill="#0f172a">{analysis.outputEquation.signal}</text>
      </svg>
    </div>
  );
}

function CircuitDiagramDeterministicUnused({
  analysis,
  flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }

  const inputName = analysis.variableNames.at(-1) ?? "X";
  const ff1 = { x: 600, y: 150, w: 90, h: 130, bit: "1" };
  const ff0 = { x: 810, y: 215, w: 90, h: 130, bit: "0" };
  const pins = getTraditionalPins(ff1, ff0, flipFlopType);
  const feedback = getTraditionalFeedback(ff1, ff0);
  const inputEquations = analysis.equations.filter((equation) => pins[equation.signal]);

  return (
    <div className="h-full min-h-[620px] overflow-x-auto rounded-lg border border-gray-300 bg-white p-4 shadow-sm">
      <svg viewBox="0 0 1120 640" className="h-full w-full min-w-[1120px]">
        <defs>
          <marker id="traditional-output-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a" />
          </marker>
        </defs>

        <rect x="0" y="0" width="1120" height="640" fill="#ffffff" />
        <text x="34" y="34" fill="#0f172a" fontSize="18" fontWeight="800">Traditional EDA Sequential Circuit Schematic</text>

        <line x1="44" y1="72" x2="1040" y2="72" stroke="#0f172a" strokeWidth="2" />
        <text x="18" y="78" fill="#0f172a" fontSize="18" fontWeight="800">{inputName}</text>
        <line x1="44" y1="578" x2="1040" y2="578" stroke="#0f172a" strokeWidth="2" />
        <text x="12" y="584" fill="#0f172a" fontSize="18" fontWeight="800">CLK</text>

        <TraditionalFeedbackLine data={feedback.Q1} />
        <TraditionalFeedbackLine data={feedback["Q1'"]} />
        <TraditionalFeedbackLine data={feedback.Q0} />
        <TraditionalFeedbackLine data={feedback["Q0'"]} />

        {inputEquations.map((equation, index) => (
          <TraditionalEquationRoute
            key={equation.signal}
            equation={equation}
            pin={pins[equation.signal]}
            inputName={inputName}
            feedback={feedback}
            inputTapX={150 + index * 26}
            gateX={pins[equation.signal].x - 126}
            laneShift={(index % 2) * 8}
          />
        ))}

        <TraditionalOutputRoute
          equation={analysis.outputEquation}
          inputName={inputName}
          feedback={feedback}
        />

        <TraditionalFlipFlop x={ff1.x} y={ff1.y} bit="1" type={flipFlopType} />
        <TraditionalFlipFlop x={ff0.x} y={ff0.y} bit="0" type={flipFlopType} />

        <line x1={ff1.x + ff1.w} y1={ff1.y + 30} x2="1040" y2={ff1.y + 30} stroke="#0f172a" strokeWidth="2" markerEnd="url(#traditional-output-arrow)" />
        <text x="1052" y={ff1.y + 36} fill="#0f172a" fontSize="18" fontWeight="800">Q1</text>
        <line x1={ff0.x + ff0.w} y1={ff0.y + 30} x2="1040" y2={ff0.y + 30} stroke="#0f172a" strokeWidth="2" markerEnd="url(#traditional-output-arrow)" />
        <text x="1052" y={ff0.y + 36} fill="#0f172a" fontSize="18" fontWeight="800">Q0</text>

        <line x1={ff1.x + 45} y1={ff1.y + ff1.h} x2={ff1.x + 45} y2="578" stroke="#0f172a" strokeWidth="2" />
        <line x1={ff0.x + 45} y1={ff0.y + ff0.h} x2={ff0.x + 45} y2="578" stroke="#0f172a" strokeWidth="2" />
        <circle cx={ff1.x + 45} cy="578" r="4" fill="#0f172a" />
        <circle cx={ff0.x + 45} cy="578" r="4" fill="#0f172a" />
      </svg>
    </div>
  );
}

function CircuitDiagramPlaUnused({
  analysis,
  flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }

  const inputName = analysis.variableNames.at(-1) ?? "X";
  const rails = getPlaRails(inputName);
  const equations = [...analysis.equations, analysis.outputEquation];
  const ff1 = { x: 560, y: 170, w: 82, h: 128, bit: "1" };
  const ff0 = { x: 780, y: 170, w: 82, h: 128, bit: "0" };
  const ffPins = getPlaPins(ff1, ff0, flipFlopType);
  const routeEquations = equations.filter((equation) => ffPins[equation.signal]);
  const outputEquation = analysis.outputEquation;

  return (
    <div className="h-full min-h-[620px] overflow-x-auto rounded-lg border border-gray-300 bg-white p-4 shadow-sm">
      <svg viewBox="0 0 1120 660" className="h-full w-full min-w-[1120px]">
        <defs>
          <marker id="pla-output-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a" />
          </marker>
        </defs>

        <rect x="0" y="0" width="1120" height="660" fill="#ffffff" />
        <text x="34" y="34" fill="#0f172a" fontSize="18" fontWeight="800">Dynamic PLA-Style Sequential Circuit</text>

        {Object.entries(rails).map(([name, rail]) => (
          <g key={name}>
            <line x1={rail.x} y1="70" x2={rail.x} y2="590" stroke="#0f172a" strokeWidth="2" />
            <text x={rail.x} y="54" textAnchor="middle" fill="#0f172a" fontSize="14" fontWeight="800">{rail.label}</text>
          </g>
        ))}

        <line x1="28" y1="122" x2={rails[inputName]?.x ?? rails.X.x} y2="122" stroke="#0f172a" strokeWidth="2" />
        <text x="12" y="127" fill="#0f172a" fontSize="16" fontWeight="800">{inputName}</text>
        <circle cx={rails[inputName]?.x ?? rails.X.x} cy="122" r="4" fill="#0f172a" />
        <line x1="28" y1="602" x2="1000" y2="602" stroke="#0f172a" strokeWidth="2" />
        <text x="10" y="607" fill="#0f172a" fontSize="16" fontWeight="800">CLK</text>

        <PlaFeedback pinX={ff1.x + ff1.w} pinY={ff1.y + 28} railX={rails.Q1.x} laneY={94} />
        <PlaFeedback pinX={ff1.x + ff1.w} pinY={ff1.y + 96} railX={rails["Q1'"].x} laneY={110} />
        <PlaFeedback pinX={ff0.x + ff0.w} pinY={ff0.y + 28} railX={rails.Q0.x} laneY={138} />
        <PlaFeedback pinX={ff0.x + ff0.w} pinY={ff0.y + 96} railX={rails["Q0'"].x} laneY={154} />

        {routeEquations.map((equation, index) => (
          <PlaEquationRoute
            key={equation.signal}
            equation={equation}
            rails={rails}
            pin={ffPins[equation.signal]}
            gateX={340 + (index % 2) * 78}
            laneOffset={index * 7}
          />
        ))}

        <PlaEquationRoute
          equation={outputEquation}
          rails={rails}
          pin={{ x: 912, y: 488, label: outputEquation.signal }}
          gateX={760}
          laneOffset={42}
        />

        <PlaFlipFlopSymbol x={ff1.x} y={ff1.y} bit="1" type={flipFlopType} />
        <PlaFlipFlopSymbol x={ff0.x} y={ff0.y} bit="0" type={flipFlopType} />

        <line x1={ff1.x + ff1.w} y1={ff1.y + 28} x2="1040" y2={ff1.y + 28} stroke="#0f172a" strokeWidth="2" markerEnd="url(#pla-output-arrow)" />
        <text x="1052" y={ff1.y + 34} fill="#0f172a" fontSize="18" fontWeight="800">Q1</text>
        <line x1={ff0.x + ff0.w} y1={ff0.y + 28} x2="1040" y2={ff0.y + 28} stroke="#0f172a" strokeWidth="2" markerEnd="url(#pla-output-arrow)" />
        <text x="1052" y={ff0.y + 34} fill="#0f172a" fontSize="18" fontWeight="800">Q0</text>
        <line x1="912" y1="488" x2="1040" y2="488" stroke="#0f172a" strokeWidth="2" markerEnd="url(#pla-output-arrow)" />
        <text x="1052" y="494" fill="#0f172a" fontSize="18" fontWeight="800">{outputEquation.signal}</text>

        <line x1="602" y1="602" x2="602" y2={ff1.y + ff1.h} stroke="#0f172a" strokeWidth="2" />
        <line x1="822" y1="602" x2="822" y2={ff0.y + ff0.h} stroke="#0f172a" strokeWidth="2" />
        <circle cx="602" cy="602" r="4" fill="#0f172a" />
        <circle cx="822" cy="602" r="4" fill="#0f172a" />
      </svg>
    </div>
  );
}

function CircuitDiagramStaticUnused({
  analysis,
  flipFlopType: _flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }
  const inputName = analysis.variableNames.at(-1) ?? "X";
  return (
    <div className="h-full min-h-[620px] overflow-x-auto rounded-lg border border-gray-300 bg-white p-4 shadow-sm">
      <svg viewBox="0 0 800 600" className="h-full w-full min-w-[800px]">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a" />
          </marker>
        </defs>

        <line x1="50" y1="50" x2="720" y2="50" stroke="#0f172a" strokeWidth="2" />
        <text x="20" y="55" fontSize="18" fontWeight="bold" fill="#0f172a">{inputName}</text>
        <line x1="50" y1="550" x2="720" y2="550" stroke="#0f172a" strokeWidth="2" />
        <text x="10" y="555" fontSize="18" fontWeight="bold" fill="#0f172a">CLK</text>

        <rect x="300" y="200" width="80" height="120" fill="none" stroke="#0f172a" strokeWidth="2" />
        <text x="310" y="230" fontSize="16" fill="#0f172a">J1</text>
        <text x="310" y="310" fontSize="16" fill="#0f172a">K1</text>
        <text x="350" y="230" fontSize="16" fill="#0f172a">Q1</text>
        <text x="345" y="310" fontSize="16" fill="#0f172a">Q1'</text>
        <polyline points="330,320 340,310 350,320" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="340" y1="320" x2="340" y2="550" stroke="#0f172a" strokeWidth="2" />
        <circle cx="340" cy="550" r="4" fill="#0f172a" />

        <rect x="550" y="200" width="80" height="120" fill="none" stroke="#0f172a" strokeWidth="2" />
        <text x="560" y="230" fontSize="16" fill="#0f172a">J0</text>
        <text x="560" y="310" fontSize="16" fill="#0f172a">K0</text>
        <text x="600" y="230" fontSize="16" fill="#0f172a">Q0</text>
        <text x="595" y="310" fontSize="16" fill="#0f172a">Q0'</text>
        <polyline points="580,320 590,310 600,320" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="590" y1="320" x2="590" y2="550" stroke="#0f172a" strokeWidth="2" />
        <circle cx="590" cy="550" r="4" fill="#0f172a" />

        <path d="M 150 230 L 170 230 A 30 30 0 0 1 170 290 L 150 290 Z" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="150" y1="230" x2="150" y2="290" stroke="#0f172a" strokeWidth="2" />
        <text x="160" y="265" fontSize="14" fontWeight="bold" fill="#0f172a">AND</text>

        <line x1="100" y1="50" x2="100" y2="245" stroke="#0f172a" strokeWidth="2" />
        <circle cx="100" cy="50" r="4" fill="#0f172a" />
        <line x1="100" y1="245" x2="150" y2="245" stroke="#0f172a" strokeWidth="2" />
        <polyline points="630,225 680,225 680,120 120,120 120,275 150,275" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="630" cy="225" r="4" fill="#0f172a" />
        <line x1="200" y1="260" x2="250" y2="260" stroke="#0f172a" strokeWidth="2" />
        <polyline points="250,260 250,225 300,225" fill="none" stroke="#0f172a" strokeWidth="2" />
        <polyline points="250,260 250,305 300,305" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="250" cy="260" r="4" fill="#0f172a" />

        <line x1="480" y1="50" x2="480" y2="225" stroke="#0f172a" strokeWidth="2" />
        <circle cx="480" cy="50" r="4" fill="#0f172a" />
        <line x1="480" y1="225" x2="550" y2="225" stroke="#0f172a" strokeWidth="2" />
        <line x1="480" y1="225" x2="480" y2="305" stroke="#0f172a" strokeWidth="2" />
        <line x1="480" y1="305" x2="550" y2="305" stroke="#0f172a" strokeWidth="2" />
        <circle cx="480" cy="225" r="4" fill="#0f172a" />

        <path d="M 600 420 L 620 420 A 30 30 0 0 1 620 480 L 600 480 Z" fill="none" stroke="#0f172a" strokeWidth="2" />
        <line x1="600" y1="420" x2="600" y2="480" stroke="#0f172a" strokeWidth="2" />
        <text x="610" y="455" fontSize="14" fontWeight="bold" fill="#0f172a">AND</text>
        <polyline points="500,50 500,435 600,435" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="500" cy="50" r="4" fill="#0f172a" />
        <polyline points="400,225 400,450 600,450" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="400" cy="225" r="4" fill="#0f172a" />
        <polyline points="650,225 650,465 600,465" fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx="650" cy="225" r="4" fill="#0f172a" />

        <line x1="380" y1="225" x2="750" y2="225" stroke="#0f172a" strokeWidth="2" markerEnd="url(#arrow)" />
        <text x="765" y="230" fontSize="18" fontWeight="bold" fill="#0f172a">Q1</text>
        <polyline points="630,245 700,245 700,255 750,255" fill="none" stroke="#0f172a" strokeWidth="2" markerEnd="url(#arrow)" />
        <text x="765" y="260" fontSize="18" fontWeight="bold" fill="#0f172a">Q0</text>
        <line x1="650" y1="450" x2="750" y2="450" stroke="#0f172a" strokeWidth="2" markerEnd="url(#arrow)" />
        <text x="765" y="455" fontSize="18" fontWeight="bold" fill="#0f172a">{analysis.outputEquation.signal}</text>
      </svg>
    </div>
  );
}

function CircuitDiagramCoordinateUnused({
  analysis,
  flipFlopType: _flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }
  const inputName = analysis.variableNames.at(-1) ?? "X";
  return (
    <div className="h-full min-h-[620px] overflow-hidden rounded-xl border border-slate-200 bg-white">
      <svg viewBox="0 0 920 640" className="h-full w-full">
        <rect x="0" y="0" width="920" height="640" fill="#ffffff" />
        <text x="34" y="36" fill="#111827" fontSize="18" fontWeight="800">
          Standard EDA Sequential Circuit Schematic
        </text>

        <defs>
          <marker id="eda-output-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#111827" />
          </marker>
        </defs>

        <text x="28" y="90" fill="#111827" fontSize="16" fontWeight="800">{inputName}</text>
        <text x="24" y="134" fill="#111827" fontSize="16" fontWeight="800">{inputName}'</text>
        <text x="24" y="580" fill="#111827" fontSize="16" fontWeight="800">CLK</text>

        <path d="M 64 84 H 854" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#eda-output-arrow)" />
        <path d="M 112 84 V 520" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 64 128 H 82" fill="none" stroke="#111827" strokeWidth="2" />
        <EdaNotGate x={82} y={112} />
        <path d="M 124 128 H 168" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 168 128 V 512" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 64 574 H 854" fill="none" stroke="#111827" strokeWidth="2" />

        <text x="862" y="90" fill="#111827" fontSize="16" fontWeight="800">Q1</text>

        <EdaAndGate x={216} y={194} width={66} height={44} />
        <path d="M 112 206 H 216" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 722 330 H 740 V 226 H 216" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 282 216 H 350" fill="none" stroke="#111827" strokeWidth="2" />

        <EdaOrGate x={214} y={352} width={78} height={54} />
        <path d="M 168 368 H 214" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 472 252 V 392 H 214" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 292 379 H 612" fill="none" stroke="#111827" strokeWidth="2" />

        <EdaAndGate x={562} y={194} width={66} height={44} />
        <path d="M 112 172 H 538 V 206 H 562" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 472 252 H 520 V 226 H 562" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 628 216 H 646" fill="none" stroke="#111827" strokeWidth="2" />

        <EdaAndGate x={738} y={466} width={78} height={56} />
        <path d="M 112 480 H 738" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 472 252 V 498 H 738" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 722 252 V 516 H 738" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 816 494 H 854" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#eda-output-arrow)" />
        <text x="862" y="500" fill="#111827" fontSize="16" fontWeight="800">{analysis.outputEquation.signal}</text>

        <path d="M 472 252 H 646" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 472 252 V 84" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 722 252 H 854" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#eda-output-arrow)" />
        <text x="862" y="258" fill="#111827" fontSize="16" fontWeight="800">Q0</text>

        <path d="M 380 574 V 300" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 676 574 V 300" fill="none" stroke="#111827" strokeWidth="2" />

        <EdaJKFlipFlop x={350} y={180} bit="1" />
        <EdaJKFlipFlop x={646} y={180} bit="0" />

        <ConnectionDot x={112} y={84} />
        <ConnectionDot x={112} y={172} />
        <ConnectionDot x={112} y={206} />
        <ConnectionDot x={112} y={480} />
        <ConnectionDot x={168} y={128} />
        <ConnectionDot x={168} y={368} />
        <ConnectionDot x={472} y={252} />
        <ConnectionDot x={472} y={498} />
        <ConnectionDot x={722} y={252} />
        <ConnectionDot x={722} y={330} />
        <ConnectionDot x={722} y={516} />
        <ConnectionDot x={380} y={574} />
        <ConnectionDot x={676} y={574} />
      </svg>
      <div className="border-t border-slate-200 bg-cyan-50 px-4 py-3 text-sm text-cyan-950">
        Diagram is generated from the simplified equations above. Change the table and press Generate again to resynthesize.
      </div>
    </div>
  );
}

function CircuitDiagramOld({
  analysis,
  flipFlopType: _flipFlopType,
}: {
  analysis: AnalysisResult | null;
  flipFlopType: FlipFlopType;
}) {
  if (!analysis) {
    return (
      <div className="grid h-full min-h-[620px] place-items-center rounded-xl border-2 border-dashed border-cyan-300 bg-gradient-to-br from-white via-cyan-50 to-emerald-50 px-6 text-center">
        <div>
          <p className="text-base font-bold text-slate-700">
            Circuit Diagram Rendering Area
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Press Generate to render flip-flops, input equations, and output logic.
          </p>
        </div>
      </div>
    );
  }
  const inputName = analysis.variableNames.at(-1) ?? "X";
  return (
    <div className="h-full min-h-[620px] overflow-hidden rounded-xl border border-slate-200 bg-white">
      <svg viewBox="0 0 920 640" className="h-full w-full">
        <rect x="0" y="0" width="920" height="640" fill="#ffffff" />
        <text x="34" y="36" fill="#111827" fontSize="18" fontWeight="800">
          Standard EDA Sequential Circuit Schematic
        </text>

        <path d="M 96 72 V 520" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 158 116 V 520" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 54 576 H 850" fill="none" stroke="#111827" strokeWidth="2" />
        <text x="26" y="77" fill="#111827" fontSize="16" fontWeight="800">
          {inputName}
        </text>
        <text x="24" y="121" fill="#111827" fontSize="16" fontWeight="800">
          {inputName}'
        </text>
        <text x="16" y="581" fill="#111827" fontSize="16" fontWeight="800">
          CLK
        </text>

        <path d="M 54 116 H 78" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M78 100 L78 132 L110 116 Z" fill="#ffffff" stroke="#111827" strokeWidth="2" />
        <circle cx="118" cy="116" r="6" fill="#ffffff" stroke="#111827" strokeWidth="2" />
        <path d="M 124 116 H 158" fill="none" stroke="#111827" strokeWidth="2" />

        <path d="M 96 72 H 850" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />
        <text x="858" y="77" fill="#111827" fontSize="16" fontWeight="800">
          Q1
        </text>

        <path d="M 96 206 H 216 V 206 H 270" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 742 292 H 754 V 438 H 216 V 226 H 270" fill="none" stroke="#111827" strokeWidth="2" />
        <AndGate x={270} y={196} />
        <path d="M 340 216 H 392" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />

        <path d="M 742 216 H 850" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />
        <text x="858" y="221" fill="#111827" fontSize="16" fontWeight="800">
          Q0
        </text>
        <path d="M 742 216 H 770 V 504" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 770 504 H 374 V 292" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 374 292 H 392" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />

        <path d="M 96 154 H 506 V 206 H 526" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 512 216 H 542 V 226 H 526" fill="none" stroke="#111827" strokeWidth="2" />
        <AndGate x={526} y={196} />
        <path d="M 596 216 H 622" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />

        <path d="M 158 370 H 506 V 280 H 526" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 512 216 V 304 H 526" fill="none" stroke="#111827" strokeWidth="2" />
        <OrGate x={526} y={268} />
        <path d="M 592 292 H 622" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />

        <path d="M 96 474 H 690" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 512 216 V 494 H 690" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 770 504 H 690" fill="none" stroke="#111827" strokeWidth="2" />
        <ThreeInputAndGate x={690} y={454} />
        <path d="M 772 484 H 850" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />
        <text x="858" y="489" fill="#111827" fontSize="16" fontWeight="800">
          {analysis.outputEquation.signal}
        </text>

        <path d="M 512 216 H 622" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />
        <path d="M 512 216 V 72 H 850" fill="none" stroke="#111827" strokeWidth="2" markerEnd="url(#arrow)" />
        <path d="M 512 216 V 494" fill="none" stroke="#111827" strokeWidth="2" />

        <path d="M 452 576 V 342" fill="none" stroke="#111827" strokeWidth="2" />
        <path d="M 682 576 V 342" fill="none" stroke="#111827" strokeWidth="2" />

        <JKStandardSymbol x={392} y={150} name="1A" />
        <JKStandardSymbol x={622} y={150} name="1B" />

        <ConnectionDot x={96} y={72} />
        <ConnectionDot x={96} y={154} />
        <ConnectionDot x={96} y={210} />
        <ConnectionDot x={96} y={474} />
        <ConnectionDot x={158} y={116} />
        <ConnectionDot x={158} y={370} />
        <ConnectionDot x={512} y={216} />
        <ConnectionDot x={512} y={494} />
        <ConnectionDot x={742} y={216} />
        <ConnectionDot x={742} y={292} />
        <ConnectionDot x={770} y={504} />
        <ConnectionDot x={452} y={576} />
        <ConnectionDot x={682} y={576} />

        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#0f172a" />
          </marker>
        </defs>
      </svg>
      <div className="border-t border-slate-200 bg-cyan-50 px-4 py-3 text-sm text-cyan-950">
        Diagram is generated from the simplified equations above. Change the table and press Generate again to resynthesize.
      </div>
    </div>
  );
}

function FlipFlopBlock({
  x,
  y,
  label,
  inputLabels,
}: {
  x: number;
  y: number;
  label: string;
  inputLabels: string[];
}) {
  const topInput = inputLabels[0] ?? "IN0";
  const bottomInput = inputLabels[1] ?? "IN1";
  return (
    <g>
      <rect
        x={x}
        y={y}
        width="120"
        height="170"
        fill="#ffffff"
        stroke="#111827"
        strokeWidth="2"
      />
      <rect
        x={x + 22}
        y={y + 48}
        width="76"
        height="78"
        fill="#ffffff"
        stroke="#111827"
        strokeWidth="2"
      />
      <text
        x={x + 60}
        y={y + 72}
        textAnchor="middle"
        fill="#111827"
        fontSize="14"
        fontWeight="800"
      >
        {label.split(" ")[0]}
      </text>
      <text
        x={x + 60}
        y={y + 94}
        textAnchor="middle"
        fill="#111827"
        fontSize="13"
        fontWeight="700"
      >
        Flip-Flop
      </text>
      <text x={x + 10} y={y + 67} fill="#111827" fontSize="12" fontWeight="800">
        {topInput}
      </text>
      <text x={x + 10} y={y + 132} fill="#111827" fontSize="12" fontWeight="800">
        {bottomInput}
      </text>
      <text x={x + 90} y={y + 67} fill="#111827" fontSize="12" fontWeight="800">
        Q
      </text>
      <text x={x + 87} y={y + 132} fill="#111827" fontSize="12" fontWeight="800">
        Q'
      </text>
      <path
        d={`M ${x + 54} ${y + 144} L ${x + 60} ${y + 134} L ${x + 66} ${y + 144}`}
        fill="none"
        stroke="#111827"
        strokeWidth="2"
      />
      <text x={x + 49} y={y + 162} fill="#111827" fontSize="12" fontWeight="800">
        CLK
      </text>
    </g>
  );
}

type TraditionalFeedback = {
  pinX: number;
  pinY: number;
  channelY: number;
  leftX: number;
  label: string;
};

function getTraditionalPins(
  ff1: { x: number; y: number; w: number; h: number },
  ff0: { x: number; y: number; w: number; h: number },
  flipFlopType: FlipFlopType,
): Record<string, PlaPin> {
  if (flipFlopType === "d") {
    return {
      D1: { x: ff1.x, y: ff1.y + 64, label: "D1" },
      D0: { x: ff0.x, y: ff0.y + 64, label: "D0" },
    };
  }
  if (flipFlopType === "t") {
    return {
      T1: { x: ff1.x, y: ff1.y + 64, label: "T1" },
      T0: { x: ff0.x, y: ff0.y + 64, label: "T0" },
    };
  }
  return {
    J1: { x: ff1.x, y: ff1.y + 38, label: "J1" },
    K1: { x: ff1.x, y: ff1.y + 100, label: "K1" },
    J0: { x: ff0.x, y: ff0.y + 38, label: "J0" },
    K0: { x: ff0.x, y: ff0.y + 100, label: "K0" },
  };
}

function getTraditionalFeedback(
  ff1: { x: number; y: number; w: number; h: number },
  ff0: { x: number; y: number; w: number; h: number },
): Record<string, TraditionalFeedback> {
  return {
    Q1: { pinX: ff1.x + ff1.w, pinY: ff1.y + 30, channelY: 446, leftX: 260, label: "Q1" },
    "Q1'": { pinX: ff1.x + ff1.w, pinY: ff1.y + 100, channelY: 470, leftX: 245, label: "Q1'" },
    Q0: { pinX: ff0.x + ff0.w, pinY: ff0.y + 30, channelY: 494, leftX: 230, label: "Q0" },
    "Q0'": { pinX: ff0.x + ff0.w, pinY: ff0.y + 100, channelY: 518, leftX: 215, label: "Q0'" },
  };
}

function TraditionalFeedbackLine({ data }: { data: TraditionalFeedback }) {
  return (
    <g>
      <polyline points={`${data.pinX},${data.pinY} ${data.pinX + 34},${data.pinY} ${data.pinX + 34},${data.channelY} ${data.leftX},${data.channelY}`} fill="none" stroke="#0f172a" strokeWidth="2" />
      <text x={data.leftX - 34} y={data.channelY + 5} fill="#0f172a" fontSize="12" fontWeight="800">{data.label}</text>
    </g>
  );
}

function TraditionalEquationRoute({
  equation,
  pin,
  inputName,
  feedback,
  inputTapX,
  gateX,
  laneShift,
}: {
  equation: EquationResult;
  pin: PlaPin;
  inputName: string;
  feedback: Record<string, TraditionalFeedback>;
  inputTapX: number;
  gateX: number;
  laneShift: number;
}) {
  const terms = parsePlaEquation(equation.equation);
  if (terms.length === 1 && terms[0].kind === "constant") {
    return (
      <g>
        <line x1={pin.x - 54} y1={pin.y} x2={pin.x} y2={pin.y} stroke="#0f172a" strokeWidth="2" />
        <circle cx={pin.x - 54} cy={pin.y} r="5" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
        <text x={pin.x - 78} y={pin.y + 5} fill="#0f172a" fontSize="16" fontWeight="800">{terms[0].value}</text>
      </g>
    );
  }
  const productTerms = terms.filter((term): term is { kind: "product"; literals: string[] } => term.kind === "product");
  if (productTerms.length === 1 && productTerms[0].literals.length === 1) {
    return (
      <TraditionalLiteralWire
        literal={productTerms[0].literals[0]}
        targetX={pin.x}
        targetY={pin.y}
        inputName={inputName}
        feedback={feedback}
        inputTapX={inputTapX}
      />
    );
  }
  if (productTerms.length === 1) {
    const gateHeight = Math.max(46, productTerms[0].literals.length * 18 + 20);
    const gateY = pin.y - gateHeight / 2 + laneShift;
    const outputY = gateY + gateHeight / 2;
    return (
      <g>
        <TraditionalAndInputs
          term={productTerms[0]}
          gateX={gateX}
          gateY={gateY}
          gateHeight={gateHeight}
          inputName={inputName}
          feedback={feedback}
          inputTapX={inputTapX}
        />
        <polyline points={`${gateX + 64},${outputY} ${pin.x - 20},${outputY} ${pin.x - 20},${pin.y} ${pin.x},${pin.y}`} fill="none" stroke="#0f172a" strokeWidth="2" />
      </g>
    );
  }
  const orX = pin.x - 84;
  const orY = pin.y - 36;
  return (
    <g>
      {productTerms.map((term, index) => {
        const gateHeight = Math.max(42, term.literals.length * 18 + 20);
        const gateY = pin.y - 80 + index * 56;
        const outputY = gateY + gateHeight / 2;
        const orInputY = orY + 20 + index * 22;
        return (
          <g key={`${equation.signal}-${index}`}>
            <TraditionalAndInputs term={term} gateX={gateX - 72} gateY={gateY} gateHeight={gateHeight} inputName={inputName} feedback={feedback} inputTapX={inputTapX + index * 18} />
            <polyline points={`${gateX - 8},${outputY} ${orX - 18},${outputY} ${orX - 18},${orInputY} ${orX},${orInputY}`} fill="none" stroke="#0f172a" strokeWidth="2" />
          </g>
        );
      })}
      <EdaOrGate x={orX} y={orY} width={64} height={72} />
      <polyline points={`${orX + 64},${orY + 36} ${pin.x - 20},${orY + 36} ${pin.x - 20},${pin.y} ${pin.x},${pin.y}`} fill="none" stroke="#0f172a" strokeWidth="2" />
    </g>
  );
}

function TraditionalLiteralWire({
  literal,
  targetX,
  targetY,
  inputName,
  feedback,
  inputTapX,
}: {
  literal: string;
  targetX: number;
  targetY: number;
  inputName: string;
  feedback: Record<string, TraditionalFeedback>;
  inputTapX: number;
}) {
  if (literal === inputName) {
    return (
      <g>
        <polyline points={`${inputTapX},72 ${inputTapX},${targetY} ${targetX},${targetY}`} fill="none" stroke="#0f172a" strokeWidth="2" />
        <circle cx={inputTapX} cy="72" r="4" fill="#0f172a" />
      </g>
    );
  }
  if (literal === `${inputName}'`) {
    return (
      <g>
        <EdaNotGate x={inputTapX - 20} y={88} />
        <line x1={inputTapX} y1="72" x2={inputTapX} y2="88" stroke="#0f172a" strokeWidth="2" />
        <circle cx={inputTapX} cy="72" r="4" fill="#0f172a" />
        <polyline points={`${inputTapX + 24},104 ${inputTapX + 24},${targetY} ${targetX},${targetY}`} fill="none" stroke="#0f172a" strokeWidth="2" />
      </g>
    );
  }
  const data = feedback[literal];
  if (!data) {
    return (
      <g>
        <line x1={targetX - 54} y1={targetY} x2={targetX} y2={targetY} stroke="#0f172a" strokeWidth="2" />
        <circle cx={targetX - 54} cy={targetY} r="5" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
        <text x={targetX - 78} y={targetY + 5} fill="#0f172a" fontSize="16" fontWeight="800">0</text>
      </g>
    );
  }
  const branchX = Math.max(data.leftX + 46, targetX - 150);
  return (
    <g>
      <polyline points={`${branchX},${data.channelY} ${branchX},${targetY} ${targetX},${targetY}`} fill="none" stroke="#0f172a" strokeWidth="2" />
      <circle cx={branchX} cy={data.channelY} r="4" fill="#0f172a" />
    </g>
  );
}

function TraditionalAndInputs({
  term,
  gateX,
  gateY,
  gateHeight,
  inputName,
  feedback,
  inputTapX,
}: {
  term: { literals: string[] };
  gateX: number;
  gateY: number;
  gateHeight: number;
  inputName: string;
  feedback: Record<string, TraditionalFeedback>;
  inputTapX: number;
}) {
  return (
    <g>
      <EdaAndGate x={gateX} y={gateY} width={64} height={gateHeight} />
      {term.literals.map((literal, index) => {
        const inputY = gateY + 14 + index * ((gateHeight - 28) / Math.max(1, term.literals.length - 1));
        return (
          <TraditionalLiteralWire
            key={`${literal}-${index}-${gateY}`}
            literal={literal}
            targetX={gateX}
            targetY={inputY}
            inputName={inputName}
            feedback={feedback}
            inputTapX={inputTapX + index * 18}
          />
        );
      })}
    </g>
  );
}

function TraditionalOutputRoute({
  equation,
  inputName,
  feedback,
}: {
  equation: EquationResult;
  inputName: string;
  feedback: Record<string, TraditionalFeedback>;
}) {
  const pin = { x: 930, y: 408, label: equation.signal };
  return (
    <g>
      <TraditionalEquationRoute equation={equation} pin={pin} inputName={inputName} feedback={feedback} inputTapX={420} gateX={760} laneShift={0} />
      <line x1={pin.x} y1={pin.y} x2="1040" y2={pin.y} stroke="#0f172a" strokeWidth="2" markerEnd="url(#traditional-output-arrow)" />
      <text x="1052" y={pin.y + 6} fill="#0f172a" fontSize="18" fontWeight="800">{equation.signal}</text>
    </g>
  );
}

function TraditionalFlipFlop({
  x,
  y,
  bit,
  type,
}: {
  x: number;
  y: number;
  bit: string;
  type: FlipFlopType;
}) {
  const label = type === "jk" ? "JK" : type === "d" ? "D" : "T";
  const topInput = type === "jk" ? `J${bit}` : `${label}${bit}`;
  const bottomInput = type === "jk" ? `K${bit}` : "";
  return (
    <g>
      <rect x={x} y={y} width="90" height="130" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
      <text x={x + 10} y={y + 43} fill="#0f172a" fontSize="14" fontWeight="800">{topInput}</text>
      {bottomInput ? <text x={x + 10} y={y + 105} fill="#0f172a" fontSize="14" fontWeight="800">{bottomInput}</text> : null}
      <text x={x + 58} y={y + 35} fill="#0f172a" fontSize="14" fontWeight="800">Q{bit}</text>
      <text x={x + 53} y={y + 105} fill="#0f172a" fontSize="14" fontWeight="800">Q{bit}'</text>
      <text x={x + 45} y={y + 65} textAnchor="middle" fill="#0f172a" fontSize="15" fontWeight="800">{label}</text>
      <text x={x + 45} y={y + 84} textAnchor="middle" fill="#0f172a" fontSize="12" fontWeight="700">Flip-Flop</text>
      <polygon points={`${x + 36},${y + 130} ${x + 45},${y + 118} ${x + 54},${y + 130}`} fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
    </g>
  );
}

type PlaRail = { x: number; label: string };
type PlaPin = { x: number; y: number; label: string };
type ParsedPlaTerm = { kind: "constant"; value: "0" | "1" } | { kind: "product"; literals: string[] };

function getPlaRails(inputName: string): Record<string, PlaRail> {
  return {
    VCC: { x: 54, label: "VCC (1)" },
    GND: { x: 92, label: "GND (0)" },
    [inputName]: { x: 130, label: inputName },
    [`${inputName}'`]: { x: 168, label: `${inputName}'` },
    Q1: { x: 206, label: "Q1" },
    "Q1'": { x: 244, label: "Q1'" },
    Q0: { x: 282, label: "Q0" },
    "Q0'": { x: 320, label: "Q0'" },
  };
}

function getPlaPins(
  ff1: { x: number; y: number; w: number; h: number },
  ff0: { x: number; y: number; w: number; h: number },
  flipFlopType: FlipFlopType,
): Record<string, PlaPin> {
  if (flipFlopType === "d") {
    return {
      D1: { x: ff1.x, y: ff1.y + 60, label: "D1" },
      D0: { x: ff0.x, y: ff0.y + 60, label: "D0" },
    };
  }
  if (flipFlopType === "t") {
    return {
      T1: { x: ff1.x, y: ff1.y + 60, label: "T1" },
      T0: { x: ff0.x, y: ff0.y + 60, label: "T0" },
    };
  }
  return {
    J1: { x: ff1.x, y: ff1.y + 34, label: "J1" },
    K1: { x: ff1.x, y: ff1.y + 96, label: "K1" },
    J0: { x: ff0.x, y: ff0.y + 34, label: "J0" },
    K0: { x: ff0.x, y: ff0.y + 96, label: "K0" },
  };
}

function parsePlaEquation(equation: string): ParsedPlaTerm[] {
  const normalized = equation.replaceAll("繚", "·").replaceAll("*", "·").replaceAll("&", "·").trim();
  if (normalized === "1") return [{ kind: "constant", value: "1" }];
  if (normalized === "0" || normalized === "") return [{ kind: "constant", value: "0" }];
  return normalized.split("+").map((term) => ({
    kind: "product",
    literals: term
      .trim()
      .split("繚")
      .map((literal) => literal.trim())
      .filter(Boolean),
  }));
}

function getRailForLiteral(literal: string, rails: Record<string, PlaRail>) {
  return rails[literal] ?? rails[literal.replace("'", "")] ?? rails.GND;
}

function PlaEquationRoute({
  equation,
  rails,
  pin,
  gateX,
  laneOffset,
}: {
  equation: EquationResult;
  rails: Record<string, PlaRail>;
  pin: PlaPin;
  gateX: number;
  laneOffset: number;
}) {
  const terms = parsePlaEquation(equation.equation);
  if (terms.length === 1 && terms[0].kind === "constant") {
    const rail = terms[0].value === "1" ? rails.VCC : rails.GND;
    return (
      <g>
        <line x1={rail.x} y1={pin.y} x2={pin.x} y2={pin.y} stroke="#0f172a" strokeWidth="2" />
        <circle cx={rail.x} cy={pin.y} r="4" fill="#0f172a" />
        <text x={pin.x - 32} y={pin.y - 8} fill="#0f172a" fontSize="12" fontWeight="800">{equation.signal}= {terms[0].value}</text>
      </g>
    );
  }
  if (terms.length === 1 && terms[0].kind === "product" && terms[0].literals.length === 1) {
    const rail = getRailForLiteral(terms[0].literals[0], rails);
    return (
      <g>
        <line x1={rail.x} y1={pin.y} x2={pin.x} y2={pin.y} stroke="#0f172a" strokeWidth="2" />
        <circle cx={rail.x} cy={pin.y} r="4" fill="#0f172a" />
        <text x={pin.x - 48} y={pin.y - 8} fill="#0f172a" fontSize="12" fontWeight="800">{equation.signal}= {terms[0].literals[0]}</text>
      </g>
    );
  }
  const productTerms = terms.filter((term): term is { kind: "product"; literals: string[] } => term.kind === "product");
  if (productTerms.length === 1) {
    const gateHeight = Math.max(42, productTerms[0].literals.length * 18 + 16);
    const gateY = pin.y - gateHeight / 2 + laneOffset;
    const outputY = gateY + gateHeight / 2;
    return (
      <g>
        <PlaAndProduct term={productTerms[0]} rails={rails} gateX={gateX} gateY={gateY} gateHeight={gateHeight} />
        <line x1={gateX + 64} y1={outputY} x2={pin.x - 20} y2={outputY} stroke="#0f172a" strokeWidth="2" />
        <line x1={pin.x - 20} y1={outputY} x2={pin.x - 20} y2={pin.y} stroke="#0f172a" strokeWidth="2" />
        <line x1={pin.x - 20} y1={pin.y} x2={pin.x} y2={pin.y} stroke="#0f172a" strokeWidth="2" />
      </g>
    );
  }
  const orX = Math.min(pin.x - 110, gateX + 168);
  const orY = pin.y - 34;
  return (
    <g>
      {productTerms.map((term, index) => {
        const gateHeight = Math.max(42, term.literals.length * 18 + 16);
        const gateY = pin.y - 86 + index * 58 + laneOffset;
        const outputY = gateY + gateHeight / 2;
        const orInputY = orY + 18 + index * 28;
        return (
          <g key={`${equation.signal}-${index}`}>
            <PlaAndProduct term={term} rails={rails} gateX={gateX} gateY={gateY} gateHeight={gateHeight} />
            <line x1={gateX + 64} y1={outputY} x2={orX} y2={outputY} stroke="#0f172a" strokeWidth="2" />
            <line x1={orX} y1={outputY} x2={orX} y2={orInputY} stroke="#0f172a" strokeWidth="2" />
          </g>
        );
      })}
      <EdaOrGate x={orX} y={orY} width={72} height={68} />
      <line x1={orX + 72} y1={orY + 34} x2={pin.x - 20} y2={orY + 34} stroke="#0f172a" strokeWidth="2" />
      <line x1={pin.x - 20} y1={orY + 34} x2={pin.x - 20} y2={pin.y} stroke="#0f172a" strokeWidth="2" />
      <line x1={pin.x - 20} y1={pin.y} x2={pin.x} y2={pin.y} stroke="#0f172a" strokeWidth="2" />
    </g>
  );
}

function PlaAndProduct({
  term,
  rails,
  gateX,
  gateY,
  gateHeight,
}: {
  term: { literals: string[] };
  rails: Record<string, PlaRail>;
  gateX: number;
  gateY: number;
  gateHeight: number;
}) {
  return (
    <g>
      <EdaAndGate x={gateX} y={gateY} width={64} height={gateHeight} />
      {term.literals.map((literal, index) => {
        const rail = getRailForLiteral(literal, rails);
        const inputY = gateY + 14 + index * ((gateHeight - 28) / Math.max(1, term.literals.length - 1));
        return (
          <g key={`${literal}-${index}-${gateY}`}>
            <line x1={rail.x} y1={inputY} x2={gateX} y2={inputY} stroke="#0f172a" strokeWidth="2" />
            <circle cx={rail.x} cy={inputY} r="4" fill="#0f172a" />
          </g>
        );
      })}
    </g>
  );
}

function PlaFeedback({
  pinX,
  pinY,
  railX,
  laneY,
}: {
  pinX: number;
  pinY: number;
  railX: number;
  laneY: number;
}) {
  return (
    <g>
      <polyline points={`${pinX},${pinY} ${pinX + 34},${pinY} ${pinX + 34},${laneY} ${railX},${laneY}`} fill="none" stroke="#0f172a" strokeWidth="2" />
      <circle cx={railX} cy={laneY} r="4" fill="#0f172a" />
    </g>
  );
}

function PlaFlipFlopSymbol({
  x,
  y,
  bit,
  type,
}: {
  x: number;
  y: number;
  bit: string;
  type: FlipFlopType;
}) {
  const mainLabel = type === "jk" ? "JK" : type === "d" ? "D" : "T";
  const upperInput = type === "jk" ? `J${bit}` : `${mainLabel}${bit}`;
  const lowerInput = type === "jk" ? `K${bit}` : "";
  return (
    <g>
      <rect x={x} y={y} width="82" height="128" fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
      <text x={x + 10} y={y + 39} fill="#0f172a" fontSize="14" fontWeight="800">{upperInput}</text>
      {lowerInput ? <text x={x + 10} y={y + 101} fill="#0f172a" fontSize="14" fontWeight="800">{lowerInput}</text> : null}
      <text x={x + 49} y={y + 33} fill="#0f172a" fontSize="14" fontWeight="800">Q{bit}</text>
      <text x={x + 44} y={y + 101} fill="#0f172a" fontSize="14" fontWeight="800">Q{bit}'</text>
      <text x={x + 41} y={y + 66} textAnchor="middle" fill="#0f172a" fontSize="15" fontWeight="800">{mainLabel}</text>
      <text x={x + 41} y={y + 84} textAnchor="middle" fill="#0f172a" fontSize="12" fontWeight="700">Flip-Flop</text>
      <polygon points={`${x + 32},${y + 128} ${x + 41},${y + 116} ${x + 50},${y + 128}`} fill="#ffffff" stroke="#0f172a" strokeWidth="2" />
    </g>
  );
}

function EdaJKFlipFlop({ x, y, bit }: { x: number; y: number; bit: string }) {
  return (
    <g>
      <rect x={x} y={y} width="60" height="80" fill="#ffffff" stroke="#111827" strokeWidth="2" />
      <text x={x + 10} y={y + 22} fill="#111827" fontSize="13" fontWeight="800">J{bit}</text>
      <text x={x + 10} y={y + 62} fill="#111827" fontSize="13" fontWeight="800">K{bit}</text>
      <text x={x + 42} y={y + 22} fill="#111827" fontSize="13" fontWeight="800">Q</text>
      <text x={x + 39} y={y + 62} fill="#111827" fontSize="13" fontWeight="800">Q'</text>
      <text x={x + 30} y={y + 40} textAnchor="middle" fill="#111827" fontSize="14" fontWeight="800">JK</text>
      <polygon points={`${x + 24},${y + 80} ${x + 30},${y + 70} ${x + 36},${y + 80}`} fill="#ffffff" stroke="#111827" strokeWidth="2" />
      <text x={x + 30} y={y + 102} textAnchor="middle" fill="#111827" fontSize="13" fontWeight="800">CLK</text>
    </g>
  );
}

function EdaAndGate({
  x,
  y,
  width,
  height,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
}) {
  const radius = height / 2;
  return (
    <path
      d={`M ${x} ${y} H ${x + width - radius} A ${radius} ${radius} 0 0 1 ${x + width - radius} ${y + height} H ${x} Z`}
      fill="#ffffff"
      stroke="#111827"
      strokeWidth="2"
    />
  );
}

function EdaOrGate({
  x,
  y,
  width,
  height,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
}) {
  return (
    <path
      d={`M ${x} ${y} C ${x + width * 0.34} ${y + height * 0.08}, ${x + width * 0.72} ${y + height * 0.18}, ${x + width} ${y + height / 2} C ${x + width * 0.72} ${y + height * 0.82}, ${x + width * 0.34} ${y + height * 0.92}, ${x} ${y + height} C ${x + width * 0.18} ${y + height * 0.66}, ${x + width * 0.18} ${y + height * 0.34}, ${x} ${y} Z`}
      fill="#ffffff"
      stroke="#111827"
      strokeWidth="2"
    />
  );
}

function EdaNotGate({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <polygon points={`${x},${y} ${x},${y + 32} ${x + 32},${y + 16}`} fill="#ffffff" stroke="#111827" strokeWidth="2" />
      <circle cx={x + 38} cy={y + 16} r="6" fill="#ffffff" stroke="#111827" strokeWidth="2" />
    </g>
  );
}

function AndGate({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <path
        d={`M ${x} ${y} L ${x + 38} ${y} C ${x + 70} ${y}, ${x + 70} ${y + 40}, ${x + 38} ${y + 40} L ${x} ${y + 40} Z`}
        fill="#ffffff"
        stroke="#111827"
        strokeWidth="2"
      />
    </g>
  );
}

function OrGate({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <path
        d={`M ${x} ${y} C ${x + 26} ${y + 4}, ${x + 48} ${y + 16}, ${x + 66} ${y + 24} C ${x + 48} ${y + 32}, ${x + 26} ${y + 44}, ${x} ${y + 48} C ${x + 14} ${y + 32}, ${x + 14} ${y + 16}, ${x} ${y} Z`}
        fill="#ffffff"
        stroke="#111827"
        strokeWidth="2"
      />
    </g>
  );
}

function JKStandardSymbol({ x, y, name }: { x: number; y: number; name: string }) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width="120"
        height="192"
        fill="#ffffff"
        stroke="#111827"
        strokeWidth="2"
      />
      <rect
        x={x + 24}
        y={y + 48}
        width="78"
        height="98"
        fill="#ffffff"
        stroke="#111827"
        strokeWidth="2"
      />
      <text x={x + 60} y={y + 28} textAnchor="middle" fill="#111827" fontSize="18" fontWeight="800">
        {name}
      </text>
      <text x={x + 12} y={y + 66} fill="#111827" fontSize="13" fontWeight="800">
        J{name === "1A" ? "1" : "0"}
      </text>
      <text x={x + 12} y={y + 142} fill="#111827" fontSize="13" fontWeight="800">
        K{name === "1A" ? "1" : "0"}
      </text>
      <text x={x + 86} y={y + 66} fill="#111827" fontSize="13" fontWeight="800">
        Q
      </text>
      <text x={x + 83} y={y + 142} fill="#111827" fontSize="13" fontWeight="800">
        Q'
      </text>
      <text x={x + 63} y={y + 82} textAnchor="middle" fill="#111827" fontSize="14" fontWeight="800">
        JK
      </text>
      <text x={x + 63} y={y + 102} textAnchor="middle" fill="#111827" fontSize="13" fontWeight="700">
        Flip-Flop
      </text>
      <path
        d={`M ${x + 52} ${y + 170} L ${x + 60} ${y + 156} L ${x + 68} ${y + 170}`}
        fill="none"
        stroke="#111827"
        strokeWidth="2"
      />
      <text x={x + 60} y={y + 184} textAnchor="middle" fill="#111827" fontSize="13" fontWeight="800">
        C1
      </text>
    </g>
  );
}

function ThreeInputAndGate({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <path
        d={`M ${x} ${y} L ${x + 42} ${y} C ${x + 82} ${y}, ${x + 82} ${y + 60}, ${x + 42} ${y + 60} L ${x} ${y + 60} Z`}
        fill="#ffffff"
        stroke="#111827"
        strokeWidth="2"
      />
    </g>
  );
}

function ConnectionDot({ x, y }: { x: number; y: number }) {
  return <circle cx={x} cy={y} r="4.5" fill="#111827" />;
}

function TimingDiagram({
  stateRows,
  triggerEdge,
}: {
  stateRows: StateRow[];
  triggerEdge: TriggerEdge;
}) {
  const inputSequence = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0];
  const cycles = inputSequence.length;
  const left = 64;
  const top = 36;
  const cycleW = 68;
  const halfW = cycleW / 2;
  const delay = 8;
  const width = left + cycles * cycleW + 36;
  const rows = [
    { name: "CLK", y: top + 18 },
    { name: "X", y: top + 78 },
    { name: "Q1", y: top + 138 },
    { name: "Q0", y: top + 198 },
    { name: "Z", y: top + 258 },
  ];

  const normalizedRows = stateRows
    .map((row) => ({
      presentState: row.presentState.trim().toUpperCase(),
      input: row.input.trim(),
      nextState: row.nextState.trim().toUpperCase(),
      output: row.output.trim(),
    }))
    .filter((row) => row.presentState && row.input && row.nextState && row.output);
  const initialState = normalizedRows[0]?.presentState || "00";
  const findTransition = (state: string, input: number) =>
    normalizedRows.find((row) => row.presentState === state && row.input === String(input));
  const bitAt = (state: string, bit: number) => (state.padStart(2, "0")[bit] === "1" ? 1 : 0);
  const edgeX = (cycle: number) =>
    left + cycle * cycleW + (triggerEdge === "rising" ? halfW : cycleW);

  const q1Transitions: Array<{ x: number; value: number }> = [];
  const q0Transitions: Array<{ x: number; value: number }> = [];
  let currentState = initialState;
  let currentQ1 = bitAt(currentState, 0);
  let currentQ0 = bitAt(currentState, 1);

  inputSequence.forEach((input, index) => {
    const transition = findTransition(currentState, input);
    const nextState = transition?.nextState || currentState;
    const nextQ1 = bitAt(nextState, 0);
    const nextQ0 = bitAt(nextState, 1);
    const transitionX = edgeX(index) + delay;
    if (nextQ1 !== currentQ1) q1Transitions.push({ x: transitionX, value: nextQ1 });
    if (nextQ0 !== currentQ0) q0Transitions.push({ x: transitionX, value: nextQ0 });
    currentState = nextState;
    currentQ1 = nextQ1;
    currentQ0 = nextQ0;
  });

  const valueY = (baseY: number, value: number) => baseY + (value ? -16 : 16);
  const waveformPath = (
    baseY: number,
    initialValue: number,
    transitions: Array<{ x: number; value: number }>,
  ) => {
    const endX = left + cycles * cycleW;
    let path = `M ${left} ${valueY(baseY, initialValue)}`;
    let value = initialValue;
    transitions
      .filter((transition) => transition.x > left && transition.x < endX + 1)
      .forEach((transition) => {
        path += ` H ${transition.x} V ${valueY(baseY, transition.value)}`;
        value = transition.value;
      });
    path += ` H ${endX}`;
    return path;
  };

  const xTransitions = inputSequence.slice(1).flatMap((value, index) =>
    value === inputSequence[index] ? [] : [{ x: left + (index + 1) * cycleW, value }],
  );
  const valueAt = (
    initialValue: number,
    transitions: Array<{ x: number; value: number }>,
    x: number,
  ) =>
    transitions.reduce(
      (value, transition) => (transition.x <= x ? transition.value : value),
      initialValue,
    );
  const initialQ1 = bitAt(initialState, 0);
  const initialQ0 = bitAt(initialState, 1);
  const zInitial = initialQ1 && initialQ0 && inputSequence[0] ? 1 : 0;
  const zTransitions = Array.from(
    new Set([...xTransitions, ...q1Transitions, ...q0Transitions].map((transition) => transition.x)),
  )
    .sort((a, b) => a - b)
    .reduce<Array<{ x: number; value: number }>>((transitions, x) => {
      const nextZ =
        valueAt(initialQ1, q1Transitions, x) &&
        valueAt(initialQ0, q0Transitions, x) &&
        valueAt(inputSequence[0], xTransitions, x)
          ? 1
          : 0;
      const previousZ = transitions.length ? transitions[transitions.length - 1].value : zInitial;
      return nextZ === previousZ ? transitions : [...transitions, { x, value: nextZ }];
    }, []);
  const clkTransitions = Array.from({ length: cycles * 2 }, (_, index) => ({
    x: left + index * halfW + halfW,
    value: index % 2 === 0 ? 1 : 0,
  }));
  const triggerLines = Array.from({ length: cycles }, (_, index) => edgeX(index));

  return (
    <section className="flex min-h-0 flex-col rounded-[4px] border border-neutral-200 bg-white p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-neutral-600">
          Timing Diagram Viewer
        </h3>
        <div className="rounded-[3px] border border-neutral-200 bg-neutral-50 px-3 py-2 text-xs font-semibold text-neutral-700">
          {triggerEdge === "rising" ? "Rising Edge 上緣" : "Falling Edge 下緣"}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-x-auto overflow-y-hidden rounded-[4px] border border-neutral-200 bg-neutral-50">
        <svg viewBox={`0 0 ${width} 340`} className="h-full min-h-[320px] min-w-[800px]">
          <rect x="0" y="0" width={width} height="340" fill="#f8fafc" />
          {triggerLines.map((x) => (
            <line key={x} x1={x} y1="24" x2={x} y2="316" stroke="#94a3b8" strokeWidth="1.4" strokeDasharray="4 5" />
          ))}
          {Array.from({ length: cycles + 1 }, (_, index) => (
            <text key={index} x={left + index * cycleW} y="326" textAnchor="middle" fontSize="11" fill="#64748b">
              {index}
            </text>
          ))}
          {rows.map((row) => (
            <g key={row.name}>
              <text x="34" y={row.y + 5} textAnchor="end" fontSize="13" fontWeight="800" fill="#0f172a">
                {row.name}
              </text>
              <line x1={left} y1={row.y + 16} x2={left + cycles * cycleW} y2={row.y + 16} stroke="#e2e8f0" strokeWidth="1" />
              <line x1={left} y1={row.y - 16} x2={left + cycles * cycleW} y2={row.y - 16} stroke="#e2e8f0" strokeWidth="1" />
            </g>
          ))}
          <path d={waveformPath(rows[0].y, 0, clkTransitions)} fill="none" stroke="#0f172a" strokeWidth="2.4" strokeLinejoin="miter" />
          <path d={waveformPath(rows[1].y, inputSequence[0], xTransitions)} fill="none" stroke="#2563eb" strokeWidth="2.4" strokeLinejoin="miter" />
          <path d={waveformPath(rows[2].y, initialQ1, q1Transitions)} fill="none" stroke="#059669" strokeWidth="2.4" strokeLinejoin="miter" />
          <path d={waveformPath(rows[3].y, initialQ0, q0Transitions)} fill="none" stroke="#7c3aed" strokeWidth="2.4" strokeLinejoin="miter" />
          <path d={waveformPath(rows[4].y, zInitial, zTransitions)} fill="none" stroke="#dc2626" strokeWidth="2.4" strokeLinejoin="miter" />
          <text x={left} y="18" fontSize="11" fontWeight="700" fill="#64748b">
            Delay: +{delay}px after selected clock edge
          </text>
        </svg>
      </div>
    </section>
  );
}

export default App;
