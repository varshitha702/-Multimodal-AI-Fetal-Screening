import { dataUrl, formatGa, type Prediction } from "@/lib/api";

interface Props {
  result: Prediction;
  originalUrl: string;
  fileName: string;
}

/** Report layout that is only visible when printing (Ctrl+P / "Print report" -> Save as PDF). */
export function PrintReport({ result, originalUrl, fileName }: Props) {
  const e = result.ellipse;
  const rows: [string, string][] = [
    ["Image", `${fileName} (${result.image_width} × ${result.image_height} px)`],
    ["Pixel spacing", result.pixel_size_mm ? `${result.pixel_size_mm} mm/px` : "not provided"],
    [
      "Plane check",
      `${result.plane_check.status.toUpperCase()}${result.plane_check.overridden ? " (overridden)" : ""}: ${result.plane_check.message}`,
    ],
    ["Head detected", result.head_detected ? "yes" : "no"],
    [
      "Head circumference",
      result.head_circumference_mm !== null
        ? `${result.head_circumference_mm.toFixed(1)} mm${result.uncertainty?.hc_sd_mm != null ? ` ± ${result.uncertainty.hc_sd_mm.toFixed(1)} mm (SD over ${result.uncertainty.n_passes} TTA passes)` : ""}`
        : "—",
    ],
    ...(result.uncertainty
      ? ([
          [
            "Reliability",
            result.uncertainty.low_reliability
              ? `LOW, re-scan recommended: ${result.uncertainty.reasons.join(" ")}`
              : `OK (CV ${result.uncertainty.hc_cv_percent?.toFixed(2) ?? "—"}%, disagreement ${(result.uncertainty.disagreement_fraction * 100).toFixed(1)}%)`,
          ],
        ] as [string, string][])
      : []),
    [
      "Ellipse diameters",
      e?.semi_major_mm ? `${(2 * e.semi_major_mm).toFixed(1)} × ${(2 * e.semi_minor_mm!).toFixed(1)} mm` : "—",
    ],
    ["Head area", result.head_area_mm2 !== null ? `${(result.head_area_mm2 / 100).toFixed(2)} cm²` : "—"],
    ...(result.growth
      ? ([
          ["Gestational age", `${formatGa(result.growth.gestational_age_weeks)} weeks`],
          [
            "HC centile (INTERGROWTH-21st)",
            `${result.growth.centile.toFixed(1)} (z ${result.growth.z_score.toFixed(2)}); median ${result.growth.median_mm.toFixed(1)} mm, 3rd–97th ${result.growth.p3_mm.toFixed(1)}–${result.growth.p97_mm.toFixed(1)} mm`,
          ],
        ] as [string, string][])
      : []),
    ["Model confidence", `${(result.quality.mean_confidence * 100).toFixed(1)}%`],
    ["Model version", result.model_version],
  ];
  return (
    <article className="hidden text-black print:block" data-testid="print-report">
      <header className="mb-4 flex items-end justify-between border-b-2 border-black pb-2">
        <div>
          <h1 className="text-2xl font-bold">FetalAI: head circumference report</h1>
          <p className="text-sm">HC18 U-Net segmentation · research prototype</p>
        </div>
        <p className="text-sm">{new Date().toLocaleString()}</p>
      </header>
      <p className="mb-4 border border-black p-2 text-sm font-semibold">{result.disclaimer}</p>
      <div className="mb-4 grid grid-cols-2 gap-4">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={originalUrl} alt="Original" className="w-full border" />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={dataUrl(result.overlay_png_base64)} alt="Overlay" className="w-full border" />
      </div>
      <table className="mb-4 w-full border-collapse text-sm">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b">
              <th className="w-1/3 py-1 pr-4 text-left font-semibold">{k}</th>
              <td className="py-1">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {result.quality.flags.length > 0 && (
        <>
          <h2 className="font-semibold">Quality warnings</h2>
          <ul className="mb-4 list-disc pl-5 text-sm">
            {result.quality.flags.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </>
      )}
      <p className="text-xs">
        Method: U-Net segmentation, ellipse fit, perimeter by Ramanujan&apos;s approximation × pixel spacing. Trained on
        the HC18 dataset (van den Heuvel et al., PLoS ONE 2018).
        {result.growth && " Centile: INTERGROWTH-21st fetal growth standard (Papageorghiou et al., Lancet 2014;384:869-79); a comparison with a population standard, not a diagnosis."}
      </p>
    </article>
  );
}
