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
  areaUnit: string
  sizeUnit: string
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
  calibrated: boolean
  appliedUmPerPx: number | null
  calibrationSource: string
}

export interface ModeBatchStats {
  imageCount: number
  totalParticleCount: number
  calibratedImageCount: number
  physicalStatsImageCount: number
  physicalStatsScope: string
  mixedCalibration: boolean
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
  physicalStatsImageNames: string[]
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

const equivalentDiameterFromArea = (areaValue: number): number =>
  areaValue > 0 ? (2 * Math.sqrt(areaValue / Math.PI)) : 0

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
  const areaUnit = String(summary.area_unit ?? 'px^2')
  const sizeUnit = String(summary.size_unit ?? summary.diameter_unit ?? 'px')
  return summary.particles.map((particle, index) => {
    const row = typeof particle === 'object' && particle ? (particle as Record<string, unknown>) : {}
    const areaValue = toNumber(row.area_value)
    return {
      imageId,
      imageName,
      label: toNumber(row.label, index + 1),
      areaPx: toNumber(row.area_px),
      areaValue,
      sizeValue: resolveNumber(
        row.equiv_diameter,
        () => resolveNumber(row.size_value, () => equivalentDiameterFromArea(areaValue)),
      ),
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
      areaUnit,
      sizeUnit,
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
        calibratedImageCount: 0,
        physicalStatsImageCount: 0,
        physicalStatsScope: 'pixels_only',
        mixedCalibration: false,
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
        physicalStatsImageNames: [],
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
    const summaryDiameters = Array.isArray(summary.diameters) ? toNumberList(summary.diameters) : []
    const channelWidthsX = toNumberList(summary.channel_widths_x)
    const channelWidthsY = toNumberList(summary.channel_widths_y)
    const sizes =
      summaryDiameters.length === areas.length && summaryDiameters.length > 0
        ? summaryDiameters
        : summarySizes.length === areas.length && summarySizes.length > 0
          ? summarySizes
          : areas.map((area) => equivalentDiameterFromArea(area))
    const particleCount = readFiniteNumber(summary.particle_count) ?? 0
    const volumeFraction = readFiniteNumber(summary.volume_fraction) ?? Number.NaN
    const meanAreaValue = resolveNumber(summary.mean_area, () => (areas.length ? mean(areas) : Number.NaN))
    const medianAreaValue = resolveNumber(summary.median_area, () => (areas.length ? median(areas) : Number.NaN))
    const stdAreaValue = resolveNumber(summary.std_area, () => (areas.length ? std(areas) : Number.NaN))
    const meanSizeValue = resolveNumber(
      summary.mean_diameter,
      () => resolveNumber(summary.mean_size, () => (sizes.length ? mean(sizes) : Number.NaN)),
    )
    const medianSizeValue = resolveNumber(
      summary.median_diameter,
      () => resolveNumber(summary.median_size, () => (sizes.length ? median(sizes) : Number.NaN)),
    )
    const stdSizeValue = resolveNumber(
      summary.std_diameter,
      () => resolveNumber(summary.std_size, () => (sizes.length ? std(sizes) : Number.NaN)),
    )
    const meanChannelWidthXValue = resolveNumber(
      summary.mean_channel_width_x,
      () => (channelWidthsX.length ? mean(channelWidthsX) : Number.NaN),
    )
    const medianChannelWidthXValue = resolveNumber(
      summary.median_channel_width_x,
      () => (channelWidthsX.length ? median(channelWidthsX) : Number.NaN),
    )
    const stdChannelWidthXValue = resolveNumber(
      summary.std_channel_width_x,
      () => (channelWidthsX.length ? std(channelWidthsX) : Number.NaN),
    )
    const meanChannelWidthYValue = resolveNumber(
      summary.mean_channel_width_y,
      () => (channelWidthsY.length ? mean(channelWidthsY) : Number.NaN),
    )
    const medianChannelWidthYValue = resolveNumber(
      summary.median_channel_width_y,
      () => (channelWidthsY.length ? median(channelWidthsY) : Number.NaN),
    )
    const stdChannelWidthYValue = resolveNumber(
      summary.std_channel_width_y,
      () => (channelWidthsY.length ? std(channelWidthsY) : Number.NaN),
    )
    const getParticles = createLazy(() => buildParticleRows(image.image_id, image.image_name, summary))

    imageRows.push({
      imageId: image.image_id,
      imageName: image.image_name,
      volumeFraction,
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
      calibrated: Boolean(summary.calibrated),
      appliedUmPerPx: readFiniteNumber(summary.applied_um_per_px),
      calibrationSource: String(summary.calibration_source ?? 'pixels_only'),
    } as ImageMetricRow)

    if (Number.isFinite(volumeFraction)) {
      volumeFractions.push(volumeFraction)
    }
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

  const backendBatch =
    payload.run.summary && typeof payload.run.summary === 'object'
      ? ((payload.run.summary as Record<string, any>).batch?.[mode] as Record<string, unknown> | undefined) ?? null
      : null
  const backendImageMeanAreas = Array.isArray(backendBatch?.image_mean_areas) ? toNumberList(backendBatch?.image_mean_areas) : null
  const backendImageMeanSizes = Array.isArray(backendBatch?.image_mean_diameters)
    ? toNumberList(backendBatch?.image_mean_diameters)
    : null
  const backendImageMeanChannelWidthsX = Array.isArray(backendBatch?.image_mean_channel_widths_x)
    ? toNumberList(backendBatch?.image_mean_channel_widths_x)
    : null
  const backendImageMeanChannelWidthsY = Array.isArray(backendBatch?.image_mean_channel_widths_y)
    ? toNumberList(backendBatch?.image_mean_channel_widths_y)
    : null
  const areaUnit =
    backendBatch?.area_unit != null ? String(backendBatch.area_unit) : (imageRows[0]?.areaUnit ?? 'px^2')
  const sizeUnit =
    backendBatch?.size_unit != null ? String(backendBatch.size_unit) : (imageRows[0]?.sizeUnit ?? 'px')
  const channelWidthUnit =
    backendBatch?.channel_width_unit != null
      ? String(backendBatch.channel_width_unit)
      : (imageRows[0]?.channelWidthUnit ?? 'px')
  const areaValuesForBatch = Array.isArray(backendBatch?.areas) ? toNumberList(backendBatch?.areas) : null
  const sizeValuesForBatch = Array.isArray(backendBatch?.diameters)
    ? toNumberList(backendBatch?.diameters)
    : null
  const channelWidthValuesXForBatch = Array.isArray(backendBatch?.channel_widths_x)
    ? toNumberList(backendBatch?.channel_widths_x)
    : null
  const channelWidthValuesYForBatch = Array.isArray(backendBatch?.channel_widths_y)
    ? toNumberList(backendBatch?.channel_widths_y)
    : null
  const areaValuesSource = areaValuesForBatch ?? areaValues
  const sizeValuesSource = sizeValuesForBatch ?? sizeValues
  const channelWidthValuesXSource = channelWidthValuesXForBatch ?? channelWidthValuesX
  const channelWidthValuesYSource = channelWidthValuesYForBatch ?? channelWidthValuesY
  const imageMeanAreasSource = backendImageMeanAreas ?? imageMeanAreas
  const imageMeanSizesSource = backendImageMeanSizes ?? imageMeanSizes
  const imageMeanChannelWidthsXSource = backendImageMeanChannelWidthsX ?? imageMeanChannelWidthsX
  const imageMeanChannelWidthsYSource = backendImageMeanChannelWidthsY ?? imageMeanChannelWidthsY
  const getVolumeFractionCi95 = createLazy(() =>
    normalizeCi(backendBatch?.volume_fraction_ci95, () => buildConfidenceInterval95(volumeFractions, 'fraction')),
  )
  const getParticleCountCi95 = createLazy(() =>
    normalizeCi(backendBatch?.particle_count_ci95, () => buildConfidenceInterval95(particleCounts, 'count')),
  )
  const getImageMeanAreaCi95 = createLazy(() =>
    normalizeCi(backendBatch?.image_mean_area_ci95, () => buildConfidenceInterval95(imageMeanAreasSource, areaUnit)),
  )
  const getImageMeanSizeCi95 = createLazy(() =>
    normalizeCi(backendBatch?.image_mean_size_ci95, () => buildConfidenceInterval95(imageMeanSizesSource, sizeUnit)),
  )
  const getImageMeanChannelWidthXCi95 = createLazy(() =>
    normalizeCi(
      backendBatch?.image_mean_channel_width_x_ci95,
      () => buildConfidenceInterval95(imageMeanChannelWidthsXSource, channelWidthUnit),
    ),
  )
  const getImageMeanChannelWidthYCi95 = createLazy(() =>
    normalizeCi(
      backendBatch?.image_mean_channel_width_y_ci95,
      () => buildConfidenceInterval95(imageMeanChannelWidthsYSource, channelWidthUnit),
    ),
  )
  const getAreaHistogram = createLazy(() => buildHistogram(areaValuesSource, 20))
  const getSizeHistogram = createLazy(() => buildHistogram(sizeValuesSource, 5))
  const getChannelWidthXHistogram = createLazy(() => buildHistogram(channelWidthValuesXSource, 12))
  const getChannelWidthYHistogram = createLazy(() => buildHistogram(channelWidthValuesYSource, 12))

  return {
    imageRows,
    get particleRows() {
      return particleRows()
    },
    batch: {
      imageCount: toNumber(backendBatch?.image_count, imageRows.length),
      totalParticleCount: toNumber(backendBatch?.particle_count_total, totalParticleCount),
      calibratedImageCount: toNumber(backendBatch?.calibrated_image_count, imageRows.filter((row) => row.calibrated).length),
      physicalStatsImageCount: toNumber(
        backendBatch?.physical_stats_image_count,
        imageRows.filter((row) => row.areaUnit === areaUnit).length,
      ),
      physicalStatsScope: String(backendBatch?.physical_stats_scope ?? 'pixels_only'),
      mixedCalibration: Boolean(backendBatch?.mixed_calibration),
      avgVolumeFraction: resolveNumber(backendBatch?.avg_volume_fraction, () => mean(volumeFractions)),
      minVolumeFraction: volumeFractions.length ? Math.min(...volumeFractions) : 0,
      maxVolumeFraction: volumeFractions.length ? Math.max(...volumeFractions) : 0,
      avgParticleCount: resolveNumber(backendBatch?.avg_particle_count, () => mean(particleCounts)),
      avgImageMeanArea: resolveNumber(backendBatch?.avg_image_mean_area, () => mean(imageMeanAreasSource)),
      avgImageMeanSize: resolveNumber(
        backendBatch?.avg_image_mean_diameter,
        () => resolveNumber(backendBatch?.avg_image_mean_size, () => mean(imageMeanSizesSource)),
      ),
      avgImageMeanChannelWidthX: resolveNumber(
        backendBatch?.avg_image_mean_channel_width_x,
        () => mean(imageMeanChannelWidthsXSource),
      ),
      avgImageMeanChannelWidthY: resolveNumber(
        backendBatch?.avg_image_mean_channel_width_y,
        () => mean(imageMeanChannelWidthsYSource),
      ),
      meanArea: resolveNumber(backendBatch?.mean_area, () => mean(areaValuesSource)),
      medianArea: resolveNumber(backendBatch?.median_area, () => median(areaValuesSource)),
      stdArea: resolveNumber(backendBatch?.std_area, () => std(areaValuesSource)),
      meanSize: resolveNumber(backendBatch?.mean_diameter, () => resolveNumber(backendBatch?.mean_size, () => mean(sizeValuesSource))),
      medianSize: resolveNumber(
        backendBatch?.median_diameter,
        () => resolveNumber(backendBatch?.median_size, () => median(sizeValuesSource)),
      ),
      stdSize: resolveNumber(backendBatch?.std_diameter, () => resolveNumber(backendBatch?.std_size, () => std(sizeValuesSource))),
      meanChannelWidthX: resolveNumber(backendBatch?.mean_channel_width_x, () => mean(channelWidthValuesXSource)),
      medianChannelWidthX: resolveNumber(backendBatch?.median_channel_width_x, () => median(channelWidthValuesXSource)),
      stdChannelWidthX: resolveNumber(backendBatch?.std_channel_width_x, () => std(channelWidthValuesXSource)),
      meanChannelWidthY: resolveNumber(backendBatch?.mean_channel_width_y, () => mean(channelWidthValuesYSource)),
      medianChannelWidthY: resolveNumber(backendBatch?.median_channel_width_y, () => median(channelWidthValuesYSource)),
      stdChannelWidthY: resolveNumber(backendBatch?.std_channel_width_y, () => std(channelWidthValuesYSource)),
      areaUnit,
      sizeUnit,
      channelWidthUnit,
      volumeFractions,
      particleCounts,
      areas: areaValuesSource,
      sizes: sizeValuesSource,
      channelWidthsX: channelWidthValuesXSource,
      channelWidthsY: channelWidthValuesYSource,
      physicalStatsImageNames: Array.isArray(backendBatch?.physical_stats_image_names)
        ? (backendBatch?.physical_stats_image_names as unknown[]).map((item) => String(item))
        : [],
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

export interface MetricFormulaVariable {
  symbolHtml: string
  meaning: string
}

export interface MetricDefinition {
  label: string
  title: string
  description: string
  formulaSvg: string
}

const wrapFormulaSvg = (viewBoxWidth: number, viewBoxHeight: number, label: string, content: string) => `
  <svg
    class="formula-svg"
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 ${viewBoxWidth} ${viewBoxHeight}"
    role="img"
    aria-label="${label}"
  >
    ${content}
  </svg>
`

export const metricDefinitions: Record<StatisticMetricKey, MetricDefinition> = {
  vf: {
    label: 'Vf',
    title: 'γ′ 相面积分数 / Vf',
    description: '统计每张图中前景像素占整图像素的比例，用作 γ′ 相面积分数（Vf）展示。',
    formulaSvg: wrapFormulaSvg(
      940,
      170,
      'Vf 等于前景像素数除以图像总像素数',
      `
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="126" y="92">V</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="160" y="112">f</text>
        <text class="formula-svg__text formula-svg__main" x="202" y="92">=</text>

        <text class="formula-svg__text formula-svg__main" x="282" y="62">Σ</text>
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="324" y="62">A</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="356" y="82">i</text>
        <line class="formula-svg__line" x1="266" y1="88" x2="412" y2="88" />
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="300" y="126">A</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="336" y="144">total</text>

        <text class="formula-svg__text formula-svg__main" x="448" y="92">=</text>

        <text class="formula-svg__text formula-svg__cn" x="538" y="62">前景像素数</text>
        <line class="formula-svg__line" x1="522" y1="88" x2="744" y2="88" />
        <text class="formula-svg__text formula-svg__cn" x="528" y="126">图像总像素数</text>
      `,
    ),
  },
  particle_count: {
    label: '颗粒数',
    title: 'γ′ 相颗粒数量',
    description: '按 8 邻域连通域统计每张图中独立 γ′ 相颗粒的数量。',
    formulaSvg: wrapFormulaSvg(
      940,
      150,
      '颗粒数等于八邻域连通域数量',
      `
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="268" y="86">N</text>
        <text class="formula-svg__text formula-svg__main" x="334" y="86">=</text>
        <text class="formula-svg__text formula-svg__cn" x="396" y="86">连通域数量（8 邻域）</text>
      `,
    ),
  },
  area: {
    label: '面积',
    title: '每个 γ′ 相面积',
    description: '统计每个 γ′ 相颗粒的面积；填写 um_per_px 后会自动换算为真实面积单位。',
    formulaSvg: wrapFormulaSvg(
      940,
      180,
      '面积等于单颗粒像素数乘以标定比例的平方，未标定时按像素数统计',
      `
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="244" y="72">A</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="280" y="92">i</text>
        <text class="formula-svg__text formula-svg__main" x="320" y="72">=</text>
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="384" y="72">N</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="422" y="92">i</text>
        <text class="formula-svg__text formula-svg__main" x="462" y="72">×</text>
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="526" y="72">s</text>
        <text class="formula-svg__text formula-svg__super formula-svg__var" x="560" y="42">2</text>

        <text class="formula-svg__text formula-svg__cn formula-svg__cn--small" x="226" y="136" text-anchor="end">未标定时：</text>
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="244" y="136">A</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="280" y="152">i</text>
        <text class="formula-svg__text formula-svg__main" x="320" y="136">=</text>
        <text class="formula-svg__text formula-svg__cn formula-svg__cn--small" x="378" y="136">像素数</text>
      `,
    ),
  },
  size: {
    label: '尺寸',
    title: '每个 γ′ 相等效直径',
    description: '将每个 γ′ 相颗粒等效为圆形后计算直径，用于表达更直观的颗粒尺寸。',
    formulaSvg: wrapFormulaSvg(
      940,
      170,
      '等效直径等于二乘以根号下面积除以圆周率',
      `
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="192" y="96">D</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="228" y="114">i</text>
        <text class="formula-svg__text formula-svg__main" x="280" y="96">=</text>
        <text class="formula-svg__text formula-svg__main" x="350" y="96">2</text>
        <path class="formula-svg__line" d="M408 100 L422 118 L442 54 H590" />
        <line class="formula-svg__line" x1="474" y1="98" x2="550" y2="98" />
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="508" y="72" text-anchor="middle">A</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="532" y="90">i</text>
        <text class="formula-svg__text formula-svg__main" x="512" y="128" text-anchor="middle">π</text>
      `,
    ),
  },
  channel_width: {
    label: '通道宽度',
    title: '水平 / 垂直 γ 通道宽度',
    description: '按水平或垂直截线统计被 γ′ 颗粒夹住的连续 γ 背景段长度，边缘开放背景不计入。',
    formulaSvg: wrapFormulaSvg(
      940,
      170,
      '水平与垂直通道宽度定义',
      `
        <text class="formula-svg__text formula-svg__main formula-svg__var" x="120" y="72">W</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="164" y="92">x</text>
        <text class="formula-svg__text formula-svg__main" x="204" y="72">=</text>
        <text class="formula-svg__text formula-svg__cn formula-svg__cn--small" x="264" y="72">水平夹持 γ 通道长度</text>

        <text class="formula-svg__text formula-svg__main formula-svg__var" x="120" y="132">W</text>
        <text class="formula-svg__text formula-svg__sub formula-svg__var" x="164" y="152">y</text>
        <text class="formula-svg__text formula-svg__main" x="204" y="132">=</text>
        <text class="formula-svg__text formula-svg__cn formula-svg__cn--small" x="264" y="132">垂直夹持 γ 通道长度</text>
      `,
    ),
  },
}

export const formatNumber = (value: number, digits = 2): string => {
  if (!Number.isFinite(value)) return '--'
  return value.toFixed(digits)
}
