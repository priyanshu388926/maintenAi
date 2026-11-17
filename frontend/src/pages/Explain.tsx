import { useState } from "react";
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { useApp, useLoad } from "../state";
import { C, Card, ErrorBox, Loading, NeedsModels, Page, Tabs } from "../ui";

type T = "global" | "cart" | "gam" | "mars";

export default function Explain() {
  const { fs, health } = useApp();
  const ready = health?.models?.[fs]?.ready;
  const g = useLoad(() => (ready ? api.explainGlobal(fs) : Promise.resolve(null)), [fs, ready]);
  const [tab, setTab] = useState<T>("global");
  const d = g.data;

  return (
    <Page title="Explainability" subtitle={`How the ${fs}-feature models reach their decisions: global feature importance, the full decision tree, GAM shape functions and MARS basis functions.`}>
      {health && !ready && <NeedsModels error={health.models[fs].error ?? ""} />}
      {g.error && <ErrorBox error={g.error} />}
      <Tabs<T> value={tab} onChange={setTab} tabs={[{ id: "global", label: "Feature importance" }, { id: "cart", label: "Decision tree rules" },
        { id: "gam", label: "GAM shape functions" }, { id: "mars", label: "MARS basis functions" }]} />
      {ready && !d && <Loading />}
      {d && tab === "global" && (
        <div className="grid g2">
          <Card title="Mean |SHAP| for XGBoost (test split)">
            <Importance data={d.shap_mean_abs} color={C.blue} />
            <p className="small muted">Average absolute contribution to the log-odds of failure across the 1,500 test machines.</p>
          </Card>
          <Card title="CART impurity importance">
            <Importance data={d.cart_importance} color={C.accent} />
            <p className="small muted">Share of the impurity reduction attributed to each feature in the pruned tree (depth {d.cart_depth}, {d.cart_leaves} leaves).</p>
          </Card>
          <Card title="Stacking meta-learner weights" className="span2">
            <Importance data={d.stacking_coef} color={C.orange} signed />
            <p className="small muted">Logistic-regression coefficients on each base model's log-odds (out-of-fold). Larger means the stack trusts that model more.</p>
          </Card>
        </div>)}
      {d && tab === "cart" && (
        <Card title={`Full CART rule set (${d.cart_depth} levels, ${d.cart_leaves} leaves)`}>
          <pre className="rules-text">{d.cart_rules}</pre>
          <p className="small muted">"class: 1" leaves predict failure. With engineered features the thresholds match the dataset's documented rules
            (power 3,500 / 9,000 W, ΔT 8.6 K, speed 1,380 rpm, wear × torque 11,000 min·Nm for type L).</p>
        </Card>)}
      {d && tab === "gam" && (
        <div className="grid g3">
          {Object.entries(d.gam_shapes).map(([name, s]) => {
            const data = s.x.map((x, i) => ({ x, y: s.y[i], band: [s.lo[i], s.hi[i]] }));
            return (
              <Card key={name} title={name}>
                <ResponsiveContainer width="100%" height={200}>
                  <ComposedChart data={data} margin={{ left: -14, right: 8 }}>
                    <CartesianGrid stroke="#eef2f6" />
                    <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} tickFormatter={v => Number(v).toFixed(0)} tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} tickFormatter={v => Number(v).toFixed(0)} />
                    <ReferenceLine y={0} stroke="#94a3b8" />
                    <Tooltip formatter={(v: number | number[]) => (Array.isArray(v) ? v.map(z => z.toFixed(2)).join(" to ") : v.toFixed(2))}
                      labelFormatter={l => `${name}: ${Number(l).toFixed(1)}`} />
                    <Area dataKey="band" stroke="none" fill={C.blue} fillOpacity={0.15} name="95% interval" />
                    <Line dataKey="y" stroke={C.blue} dot={false} strokeWidth={2} name="effect on log-odds" />
                  </ComposedChart>
                </ResponsiveContainer>
              </Card>);
          })}
          <Card title="Reading the curves">
            <p className="small">Each curve is that feature's own effect on the log-odds of failure, with the others held fixed.
              Values above zero raise risk. Shaded bands are 95% intervals; they widen where data are sparse.</p>
          </Card>
        </div>)}
      {d && tab === "mars" && (
        <Card title="Selected MARS basis functions">
          <pre className="rules-text">{Array.isArray(d.mars_terms) ? d.mars_terms.join("\n") : d.mars_terms}</pre>
          <p className="small muted">Hinge functions h(x − t) = max(0, x − t) on standardised inputs (x0 = type M, x1 = type H, then the numeric
            features in order{fs === "engineered" ? ", then temperature difference, power and wear × torque" : ""}). Coefficients are on the log-odds scale.</p>
        </Card>)}
    </Page>
  );
}

function Importance({ data, color, signed }: { data: Record<string, number>; color: string; signed?: boolean }) {
  const rows = Object.entries(data).map(([k, v]) => ({ k, v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  return (
    <ResponsiveContainer width="100%" height={Math.max(180, rows.length * 30)}>
      <BarChart data={rows} layout="vertical" margin={{ left: 0, right: 16 }}>
        <CartesianGrid horizontal={false} stroke="#eef2f6" />
        <XAxis type="number" tickFormatter={v => Number(v).toFixed(2)} />
        <YAxis type="category" dataKey="k" width={170} tick={{ fontSize: 12 }} />
        {signed && <ReferenceLine x={0} stroke="#475569" />}
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Bar dataKey="v" fill={color} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
