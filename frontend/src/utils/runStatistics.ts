import type { RunResultsPayload } from '../types'

export type StatisticMetricKey = 'vf' | 'particle_count' | 'area' | 'size' | 'channel_width'

export interface ConfidenceInterval95 {
  available: boolean
  n: number
  mean: number
  lower: number | null
  upper: number | null
  halfWidth: number | null
  criticalValue: number | null
  unit: string | null
  method: string
  reason?: string | null
}

export interface HistogramBucket {
  label: string
  start: number
  end: number
  count: number
  probability: number
}

export interface ParticleMetricRow {
  imageId: number
  imageName: string
  label: number
  areaPx: number
  areaValue: number
  sizeValue: number
  equivDiameter: number
  perimeter: number
  major: number
  minor: number
  feret: number
  minFeret: number
  aspectRatio: number
  circularity: number
  roundness: number
  solidity: number
  centroidX: number
  centroidY: number
  bboxX: number
  bboxY: number
  bboxW: number
  bboxH: number
  filtered: boolean
  filterReason: string
}

export interface ImageMetricRow {
  imageId: number
  imageName: string
  volumeFraction: number
  particleCount: number
  meanArea: number
  medianArea: number
  stdArea: number
  meanSize: number
  medianSize: number
  stdSize: number
  meanChannelWidthX: number
  medianChannelWidthX: number
  stdChannelWidthX: number
  meanChannelWidthY: number
  medianChannelWidthY: number
  stdChannelWidthY: number
  areas: number[]
  sizes: number[]
  channelWidthsX: number[]
  channelWidthsY: number[]
  particles: ParticleMetricRow[]
  areaUnit: string
  sizeUnit: string
  channelWidthUnit: string
}

export interface ModeBatchStats {
  imageCount: number
  totalParticleCount: number
  avgVolumeFraction: number
  minVolumeFraction: number
  maxVolumeFraction: number
  avgParticleCount: number
  avgImageMeanArea: number
  avgImageMeanSize: number
  avgImageMeanChannelWidthX: number
  avgImageMeanChannelWidthY: number
  meanArea: number
  medianArea: number
  stdArea: number
  meanSize: number
  medianSize: number
  stdSize: number
  meanChannelWidthX: number
  medianChannelWidthX: number
  stdChannelWidthX: number
  meanChannelWidthY: number
  medianChannelWidthY: number
  stdChannelWidthY: number
  areaUnit: string
  sizeUnit: string
  channelWidthUnit: string
  volumeFractions: number[]
  particleCounts: number[]
  areas: number[]
  sizes: number[]
  channelWidthsX: number[]
  channelWidthsY: number[]
  areaHistogram: HistogramBucket[]
  sizeHistogram: HistogramBucket[]
  channelWidthXHistogram: HistogramBucket[]
  channelWidthYHistogram: HistogramBucket[]
  volumeFractionCi95: ConfidenceInterval95
  particleCountCi95: ConfidenceInterval95
  imageMeanAreaCi95: ConfidenceInterval95
  imageMeanSizeCi95: ConfidenceInterval95
  imageMeanChannelWidthXCi95: ConfidenceInterval95
  imageMeanChannelWidthYCi95: ConfidenceInterval95
}

export interface ModeStatisticsAnalysis {
  imageRows: ImageMetricRow[]
  particleRows: ParticleMetricRow[]
  batch: ModeBatchStats
}

const readFiniteNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return null
}

const toNumber = (value: unknown, fallback = 0): number => readFiniteNumber(value) ?? fallback

const resolveNumber = (value: unknown, fallback: () => number): number => {
  const direct = readFiniteNumber(value)
  return direct ?? fallback()
}

const toNumberList = (value: unknown): number[] => {
  if (!Array.isArray(value)) return []
  return value.map((item) => toNumber(item)).filter((item) => Number.isFinite(item))
}

const mean = (values: number[]): number => {
  if (!values.length) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

const median = (values: number[]): number => {
  if (!values.length) return 0
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2
  }
  return sorted[middle]
}

const std = (values: number[]): number => {
  if (!values.length) return 0
  const avg = mean(values)
  return Math.sqrt(mean(values.map((value) => (value - avg) ** 2)))
}

const sampleStd = (values: number[]): number => {
  if (values.length < 2) return 0
  const avg = mean(values)
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1)
  return Math.sqrt(variance)
}

const T_CRITICAL_975: Record<number, number> = {
  1: 12.706,
  2: 4.303,
  3: 3.182,
  4: 2.776,
  5: 2.571,
  6: 2.447,
  7: 2.365,
  8: 2.306,
  9: 2.262,
  10: 2.228,
  11: 2.201,
  12: 2.179,
  13: 2.16,
  14: 2.145,
  15: 2.131,
  16: 2.12,
  17: 2.11,
  18: 2.101,
  19: 2.093,
  20: 2.086,
  21: 2.08,
  22: 2.074,
  23: 2.069,
  24: 2.064,
  25: 2.06,
  26: 2.056,
  27: 2.052,
  28: 2.048,
  29: 2.045,
  30: 2.042,
}

const studentTCritical95 = (degreesOfFreedom: number): number => {
  if (degreesOfFreedom <= 0) return 0
  if (degreesOfFreedom in T_CRITICAL_975) return T_CRITICAL_975[degreesOfFreedom]
  if (degreesOfFreedom <= 40) return 2.021
  if (degreesOfFreedom <= 60) return 2.0
  if (degreesOfFreedom <= 120) return 1.98
  return 1.96
}

const buildConfidenceInterval95 = (values: number[], unit: string | null): ConfidenceInterval95 => {
  const cleanValues = values.filter((item) => Number.isFinite(item))
  const n = cleanValues.length
  const sampleMean = mean(cleanValues)
  if (!n) {
    return {
      available: false,
      n: 0,
      mean: 0,
      lower: null,
      upper: null,
      halfWidth: null,
      criticalValue: null,
      unit,
      method: 'student_t',
      reason: '无有效图像级样本',
    }
  }
  if (n === 1) {
    return {
      available: false,
      n: 1,
      mean: sampleMean,
      lower: null,
      upper: null,
      halfWidth: null,
      criticalValue: null,
      unit,
      method: 'student_t',
      reason: '图像级样本数不足 2，无法估计 95% CI',
    }
  }
  const sampleDeviation = sampleStd(cleanValues)
  const criticalValue = studentTCritical95(n - 1)
  const halfWidth = criticalValue * sampleDeviation / Math.sqrt(n)
  return {
    available: true,
    n,
    mean: sampleMean,
    lower: sampleMean - halfWidth,
    upper: sampleMean + halfWidth,
    halfWidth,
    criticalValue,
    unit,
    method: 'student_t',
    reason: null,
  }
}

const normalizeCi = (value: unknown, fallback: () => ConfidenceInterval95): ConfidenceInterval95 => {
  if (!value || typeof value !== 'object') return fallback()
  const row = value as Record<string, unknown>
  let fallbackValue: ConfidenceInterval95 | null = null
  const getFallback = () => {
    if (fallbackValue === null) {
      fallbackValue = fallback()
    }
    return fallbackValue
  }
  const available = Boolean(row.available)
  return {
    available,
    n: toNumber(row.n),
    mean: resolveNumber(row.mean, () => getFallback().mean),
    lower: row.lower == null ? null : toNumber(row.lower),
    upper: row.upper == null ? null : toNumber(row.upper),
    halfWidth: row.half_width == null ? null : toNumber(row.half_width),
    criticalValue: row.critical_value == null ? null : toNumber(row.critical_value),
    unit: row.unit == null ? getFallback().unit : String(row.unit),
    method: String(row.method ?? getFallback().method),
    reason: row.reason == null ? getFallback().reason ?? null : String(row.reason),
  }
}

const buildHistogram = (values: number[], preferredBins?: number): HistogramBucket[] => {
  if (!values.length) return []
  let minValue = values[0]
  let maxValue = values[0]
  for (let index = 1; index < values.length; index += 1) {
    const value = values[index]
    if (value < minValue) minValue = value
    if (value > maxValue) maxValue = value
  }
  if (minValue === maxValue) {
    return [
      {
        label: `${minValue.toFixed(2)}`,
        start: minValue,
        end: maxValue,
        count: values.length,
        probability: 1,
      },
    ]
  }

  const autoBinCount = Math.min(14, Math.max(6, Math.round(Math.sqrt(values.length))))
  const binCount = preferredBins
    ? Math.max(1, Math.min(preferredBins, values.length))
    : autoBinCount
  const width = (maxValue - minValue) / binCount
  const counts = new Array<number>(binCount).fill(0)

  for (const value of values) {
    const rawIndex = Math.floor((value - minValue) / width)
    const index = Math.min(binCount - 1, Math.max(0, rawIndex))
    counts[index] += 1
  }

  return counts.map((count, index) => {
    const start = minValue + width * index
    const end = index === binCount - 1 ? maxValue : minValue + width * (index + 1)
    return {
      label: `${start.toFixed(2)} - ${end.toFixed(2)}`,
      start,
      end,
      count,
      probability: count / values.length,
    }
  })
}

const buildParticleRows = (imageId: number, imageName: string, summary: Record<string, unknown>): ParticleMetricRow[] => {
  if (!Array.isArray(summary.particles)) return []
  return summary.particles.map((particle, index) => {
    const row = typeof particle === 'object' && particle ? (particle as Record<string, unknown>) : {}
    return {
      imageId,
      imageName,
      label: toNumber(row.label, index + 1),
      areaPx: toNumber(row.area_px),
      areaValue: toNumber(row.area_value),
      sizeValue: resolveNumber(row.size_value, () => Math.sqrt(Math.max(toNumber(row.area_value), 0))),
      equivDiameter: toNumber(row.equiv_diameter),
      perimeter: resolveNumber(row.perimeter_value, () => toNumber(row.perimeter)),
      major: resolveNumber(row.major_value, () => toNumber(row.major)),
      minor: resolveNumber(row.minor_value, () => toNumber(row.minor)),
      feret: resolveNumber(row.feret_value, () => toNumber(row.feret)),
      minFeret: resolveNumber(row.minferet_value, () => toNumber(row.minferet)),
      aspectRatio: toNumber(row.aspect_ratio),
      circularity: toNumber(row.circularity),
      roundness: toNumber(row.roundness),
      solidity: toNumber(row.solidity),
      centroidX: toNumber(row.centroid_x),
      centroidY: toNumber(row.centroid_y),
      bboxX: toNumber(row.bbox_x),
      bboxY: toNumber(row.bbox_y),
      bboxW: toNumber(row.bbox_w),
      bboxH: toNumber(row.bbox_h),
      filtered: Boolean(row.filtered),
      filterReason: typeof row.filter_reason === 'string' ? row.filter_reason : '',
    }
  })
}

const createLazy = <T>(factory: () => T) => {
  let resolved = false
  let cached: T
  return () => {
    if (!resolved) {
      cached = factory()
      resolved = true
    }
    return cached
  }
}

export const getAvailableModes = (payload: RunResultsPayload | null): string[] => {
  if (!payload) return []
  const modeSet = new Set<string>()
  payload.images.forEach((image) => {
    Object.keys(image.modes ?? {}).forEach((mode) => modeSet.add(mode))
  })
  return Array.from(modeSet)
}

export const buildModeStatistics = (payload: RunResultsPayload | null, mode: string): ModeStatisticsAnalysis => {
  if (!payload || !mode) {
    return {
      imageRows: [],
      particleRows: [],
      batch: {
        imageCount: 0,
        totalParticleCount: 0,
        avgVolumeFraction: 0,
        minVolumeFraction: 0,
        maxVolumeFraction: 0,
        avgParticleCount: 0,
        avgImageMeanArea: 0,
        avgImageMeanSize: 0,
        avgImageMeanChannelWidthX: 0,
        avgImageMeanChannelWidthY: 0,
        meanArea: 0,
        medianArea: 0,
        stdArea: 0,
        meanSize: 0,
        medianSize: 0,
        stdSize: 0,
        meanChannelWidthX: 0,
        medianChannelWidthX: 0,
        stdChannelWidthX: 0,
        meanChannelWidthY: 0,
        medianChannelWidthY: 0,
        stdChannelWidthY: 0,
        areaUnit: 'px^2',
        sizeUnit: 'px',
        channelWidthUnit: 'px',
        volumeFractions: [],
        particleCounts: [],
        areas: [],
        sizes: [],
        channelWidthsX: [],
        channelWidthsY: [],
        areaHistogram: [],
        sizeHistogram: [],
        channelWidthXHistogram: [],
        channelWidthYHistogram: [],
        volumeFractionCi95: buildConfidenceInterval95([], 'fraction'),
        particleCountCi95: buildConfidenceInterval95([], 'count'),
        imageMeanAreaCi95: buildConfidenceInterval95([], 'px^2'),
        imageMeanSizeCi95: buildConfidenceInterval95([], 'px'),
        imageMeanChannelWidthXCi95: buildConfidenceInterval95([], 'px'),
        imageMeanChannelWidthYCi95: buildConfidenceInterval95([], 'px'),
      },
    }
  }

  const imageRows: ImageMetricRow[] = []
  const areaValues: number[] = []
  const sizeValues: number[] = []
  const channelWidthValuesX: number[] = []
  const channelWidthValuesY: number[] = []
  const volumeFractions: number[] = []
  const particleCounts: number[] = []
  const imageMeanAreas: number[] = []
  const imageMeanSizes: number[] = []
  const imageMeanChannelWidthsX: number[] = []
  const imageMeanChannelWidthsY: number[] = []
  let totalParticleCount = 0

  for (const image of payload.images) {
    const summary = image.modes?.[mode]?.summary as Record<string, unknown> | undefined
    if (!summary) continue

    const areas = toNumberList(summary.areas)
    const summarySizes = Array.isArray(summary.sizes) ? toNumberList(summary.sizes) : []
    const channelWidthsX = toNumberList(summary.channel_widths_x)
    const channelWidthsY = toNumberList(summary.channel_widths_y)
    const sizes =
      summarySizes.length === areas.length && summarySizes.length > 0
        ? summarySizes
        : areas.map((area) => Math.sqrt(Math.max(area, 0)))
    const particleCount = toNumber(summary.particle_count)
    const meanAreaValue = resolveNumber(summary.mean_area, () => mean(areas))
    const medianAreaValue = resolveNumber(summary.median_area, () => median(areas))
    const stdAreaValue = resolveNumber(summary.std_area, () => std(areas))
    const meanSizeValue = resolveNumber(summary.mean_size, () => mean(sizes))
    const medianSizeValue = resolveNumber(summary.median_size, () => median(sizes))
    const stdSizeValue = resolveNumber(summary.std_size, () => std(sizes))
    const meanChannelWidthXValue = resolveNumber(summary.mean_channel_width_x, () => mean(channelWidthsX))
    const medianChannelWidthXValue = resolveNumber(summary.median_channel_width_x, () => median(channelWidthsX))
    const stdChannelWidthXValue = resolveNumber(summary.std_channel_width_x, () => std(channelWidthsX))
    const meanChannelWidthYValue = resolveNumber(summary.mean_channel_width_y, () => mean(channelWidthsY))
    const medianChannelWidthYValue = resolveNumber(summary.median_channel_width_y, () => median(channelWidthsY))
    const stdChannelWidthYValue = resolveNumber(summary.std_channel_width_y, () => std(channelWidthsY))
    const getParticles = createLazy(() => buildParticleRows(image.image_id, image.image_name, summary))

    imageRows.push({
      imageId: image.image_id,
      imageName: image.image_name,
      volumeFraction: toNumber(summary.volume_fraction),
      particleCount,
      meanArea: meanAreaValue,
      medianArea: medianAreaValue,
      stdArea: stdAreaValue,
      meanSize: meanSizeValue,
      medianSize: medianSizeValue,
      stdSize: stdSizeValue,
      meanChannelWidthX: meanChannelWidthXValue,
      medianChannelWidthX: medianChannelWidthXValue,
      stdChannelWidthX: stdChannelWidthXValue,
      meanChannelWidthY: meanChannelWidthYValue,
      medianChannelWidthY: medianChannelWidthYValue,
      stdChannelWidthY: stdChannelWidthYValue,
      areas,
      sizes,
      channelWidthsX,
      channelWidthsY,
      get particles() {
        return getParticles()
      },
      areaUnit: String(summary.area_unit ?? 'px^2'),
      sizeUnit: String(summary.size_unit ?? summary.diameter_unit ?? 'px'),
      channelWidthUnit: String(summary.channel_width_unit ?? summary.size_unit ?? 'px'),
    } as ImageMetricRow)

    volumeFractions.push(toNumber(summary.volume_fraction))
    particleCounts.push(particleCount)
    totalParticleCount += particleCount
    areaValues.push(...areas)
    sizeValues.push(...sizes)
    channelWidthValuesX.push(...channelWidthsX)
    channelWidthValuesY.push(...channelWidthsY)
    if (particleCount > 0) {
      imageMeanAreas.push(meanAreaValue)
      imageMeanSizes.push(meanSizeValue)
    }
    if (channelWidthsX.length > 0) {
      imageMeanChannelWidthsX.push(meanChannelWidthXValue)
    }
    if (channelWidthsY.length > 0) {
      imageMeanChannelWidthsY.push(meanChannelWidthYValue)
    }
  }

  const particleRows = createLazy(() => {
    const rows: ParticleMetricRow[] = []
    for (const row of imageRows) {
      rows.push(...row.particles)
    }
    return rows
  })

  const areaUnit = imageRows[0]?.areaUnit ?? 'px^2'
  const sizeUnit = imageRows[0]?.sizeUnit ?? 'px'
  const channelWidthUnit = imageRows[0]?.channelWidthUnit ?? 'px'
  const backendBatch =
    payload.run.summary && typeof payload.run.summary === 'object'
      ? ((payload.run.summary as Record<string, any>).batch?.[mode] as Record<string, unknown> | undefined) ?? null
      : null
  const getVolumeFractionCi95 = createLazy(() =>
    normalizeCi(backendBatch?.volume_fraction_ci95, () => buildConfidenceInterval95(volumeFractions, 'fraction')),
  )
  const getParticleCountCi95 = createLazy(() =>
    normalizeCi(backendBatch?.particle_count_ci95, () => buildConfidenceInterval95(particleCounts, 'count')),
  )
  const getImageMeanAreaCi95 = createLazy(() =>
    normalizeCi(backendBatch?.image_mean_area_ci95, () => buildConfidenceInterval95(imageMeanAreas, areaUnit)),
  )
  const getImageMeanSizeCi95 = createLazy(() =>
    normalizeCi(backendBatch?.image_mean_size_ci95, () => buildConfidenceInterval95(imageMeanSizes, sizeUnit)),
  )
  const getImageMeanChannelWidthXCi95 = createLazy(() =>
    normalizeCi(
      backendBatch?.image_mean_channel_width_x_ci95,
      () => buildConfidenceInterval95(imageMeanChannelWidthsX, channelWidthUnit),
    ),
  )
  const getImageMeanChannelWidthYCi95 = createLazy(() =>
    normalizeCi(
      backendBatch?.image_mean_channel_width_y_ci95,
      () => buildConfidenceInterval95(imageMeanChannelWidthsY, channelWidthUnit),
    ),
  )
  const getAreaHistogram = createLazy(() => buildHistogram(areaValues, 20))
  const getSizeHistogram = createLazy(() => buildHistogram(sizeValues, 5))
  const getChannelWidthXHistogram = createLazy(() => buildHistogram(channelWidthValuesX, 12))
  const getChannelWidthYHistogram = createLazy(() => buildHistogram(channelWidthValuesY, 12))

  return {
    imageRows,
    get particleRows() {
      return particleRows()
    },
    batch: {
      imageCount: imageRows.length,
      totalParticleCount,
      avgVolumeFraction: resolveNumber(backendBatch?.avg_volume_fraction, () => mean(volumeFractions)),
      minVolumeFraction: volumeFractions.length ? Math.min(...volumeFractions) : 0,
      maxVolumeFraction: volumeFractions.length ? Math.max(...volumeFractions) : 0,
      avgParticleCount: resolveNumber(backendBatch?.avg_particle_count, () => mean(particleCounts)),
      avgImageMeanArea: resolveNumber(backendBatch?.avg_image_mean_area, () => mean(imageMeanAreas)),
      avgImageMeanSize: resolveNumber(backendBatch?.avg_image_mean_size, () => mean(imageMeanSizes)),
      avgImageMeanChannelWidthX: resolveNumber(
        backendBatch?.avg_image_mean_channel_width_x,
        () => mean(imageMeanChannelWidthsX),
      ),
      avgImageMeanChannelWidthY: resolveNumber(
        backendBatch?.avg_image_mean_channel_width_y,
        () => mean(imageMeanChannelWidthsY),
      ),
      meanArea: resolveNumber(backendBatch?.mean_area, () => mean(areaValues)),
      medianArea: resolveNumber(backendBatch?.median_area, () => median(areaValues)),
      stdArea: resolveNumber(backendBatch?.std_area, () => std(areaValues)),
      meanSize: resolveNumber(backendBatch?.mean_size, () => mean(sizeValues)),
      medianSize: resolveNumber(backendBatch?.median_size, () => median(sizeValues)),
      stdSize: resolveNumber(backendBatch?.std_size, () => std(sizeValues)),
      meanChannelWidthX: resolveNumber(backendBatch?.mean_channel_width_x, () => mean(channelWidthValuesX)),
      medianChannelWidthX: resolveNumber(backendBatch?.median_channel_width_x, () => median(channelWidthValuesX)),
      stdChannelWidthX: resolveNumber(backendBatch?.std_channel_width_x, () => std(channelWidthValuesX)),
      meanChannelWidthY: resolveNumber(backendBatch?.mean_channel_width_y, () => mean(channelWidthValuesY)),
      medianChannelWidthY: resolveNumber(backendBatch?.median_channel_width_y, () => median(channelWidthValuesY)),
      stdChannelWidthY: resolveNumber(backendBatch?.std_channel_width_y, () => std(channelWidthValuesY)),
      areaUnit,
      sizeUnit,
      channelWidthUnit,
      volumeFractions,
      particleCounts,
      areas: areaValues,
      sizes: sizeValues,
      channelWidthsX: channelWidthValuesX,
      channelWidthsY: channelWidthValuesY,
      get areaHistogram() {
        return getAreaHistogram()
      },
      get sizeHistogram() {
        return getSizeHistogram()
      },
      get channelWidthXHistogram() {
        return getChannelWidthXHistogram()
      },
      get channelWidthYHistogram() {
        return getChannelWidthYHistogram()
      },
      get volumeFractionCi95() {
        return getVolumeFractionCi95()
      },
      get particleCountCi95() {
        return getParticleCountCi95()
      },
      get imageMeanAreaCi95() {
        return getImageMeanAreaCi95()
      },
      get imageMeanSizeCi95() {
        return getImageMeanSizeCi95()
      },
      get imageMeanChannelWidthXCi95() {
        return getImageMeanChannelWidthXCi95()
      },
      get imageMeanChannelWidthYCi95() {
        return getImageMeanChannelWidthYCi95()
      },
    } as ModeBatchStats,
  }
}

export const metricDefinitions: Record<
  StatisticMetricKey,
  {
    label: string
    title: string
    description: string
    formula: string
  }
> = {
  vf: {
    label: 'Vf',
    title: 'γ′ 相面积分数 / Vf',
    description: '按每张图前景掩码像素占整图像素的比例统计，并在二维截面上作为体积分数近似展示。',
    formula: 'Vf = ΣA_i / A_total = 前景像素数 / 图像总像素数',
  },
  particle_count: {
    label: '颗粒数',
    title: 'γ′ 相颗粒数量',
    description: '按 8 邻域连通域计数，反映每张图中独立 γ′ 相颗粒的数量水平。',
    formula: 'N = 连通域数量（8 邻域）',
  },
  area: {
    label: '面积',
    title: '每个 γ′ 相面积',
    description: '对每个连通颗粒统计像素面积；若提供 um_per_px，则换算为真实面积单位。',
    formula: 'A_i = N_i × s²，未标定时 A_i = 像素数',
  },
  size: {
    label: '尺寸',
    title: '每个 γ′ 相等效边长',
    description: '按论文口径，由单个 γ′ 相面积进一步换算得到等效边长，用于做尺寸分布和平均尺寸统计。',
    formula: 'R_i = √A_i',
  },
  channel_width: {
    label: '通道宽度',
    title: 'γ 通道宽度',
    description: '按水平与垂直方向的扫描线统计颗粒间背景间隙宽度，适合普通块状组织的基础几何表征。',
    formula: 'W_x / W_y = 相邻 γ′ 边界之间的连续背景段长度',
  },
}

export const formatNumber = (value: number, digits = 2): string => {
  if (!Number.isFinite(value)) return '--'
  return value.toFixed(digits)
}
