import type { RunResultImage } from '../types'

export interface ParticleRow {
  imageId: number
  imageName: string
  label: number
  areaPx: number
  areaValue: number
  equivDiameter: number
  bboxX: number
  bboxY: number
  bboxW: number
  bboxH: number
  centroidX: number
  centroidY: number
}

export interface ModeImageRow {
  imageId: number
  imageName: string
  volumeFraction: number
  particleCount: number
  meanArea: number
  medianArea: number
  stdArea: number
  meanDiameter: number
  medianDiameter: number
  stdDiameter: number
  areas: number[]
  diameters: number[]
  particles: ParticleRow[]
  areaUnit: string
  diameterUnit: string
  foregroundPixels: number
  totalPixels: number
}

export interface ModeBatchSummary {
  imageCount: number
  volumeFraction: number
  avgVolumeFraction: number
  avgParticleCount: number
  particleCountTotal: number
  meanArea: number
  medianArea: number
  stdArea: number
  meanDiameter: number
  medianDiameter: number
  stdDiameter: number
  areaUnit: string
  diameterUnit: string
  areas: number[]
  diameters: number[]
  volumeFractions: number[]
  particleCounts: number[]
}

const toNumber = (value: unknown, fallback = 0) => {
  const next = Number(value)
  return Number.isFinite(next) ? next : fallback
}

const mean = (values: number[]) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0)

const median = (values: number[]) => {
  if (!values.length) return 0
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

const std = (values: number[]) => {
  if (!values.length) return 0
  const avg = mean(values)
  return Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length)
}

export const modeLabel = (mode: string) => {
  if (mode === 'traditional') return '传统分割'
  if (mode === 'dl') return '深度学习'
  return mode
}

export const getModeRows = (images: RunResultImage[], mode: string): ModeImageRow[] =>
  images
    .filter((image) => image.modes?.[mode]?.summary)
    .map((image) => {
      const summary = image.modes[mode].summary ?? {}
      const areas = Array.isArray(summary.areas) ? summary.areas.map((value: unknown) => toNumber(value)) : []
      const diameters = Array.isArray(summary.diameters) ? summary.diameters.map((value: unknown) => toNumber(value)) : []
      const particles = Array.isArray(summary.particles)
        ? summary.particles.map((particle: Record<string, unknown>) => ({
            imageId: image.image_id,
            imageName: String(summary.image_name ?? image.image_name),
            label: toNumber(particle.label),
            areaPx: toNumber(particle.area_px),
            areaValue: toNumber(particle.area_value),
            equivDiameter: toNumber(particle.equiv_diameter),
            bboxX: toNumber(particle.bbox_x),
            bboxY: toNumber(particle.bbox_y),
            bboxW: toNumber(particle.bbox_w),
            bboxH: toNumber(particle.bbox_h),
            centroidX: toNumber(particle.centroid_x),
            centroidY: toNumber(particle.centroid_y),
          }))
        : []

      return {
        imageId: image.image_id,
        imageName: String(summary.image_name ?? image.image_name),
        volumeFraction: toNumber(summary.volume_fraction),
        particleCount: toNumber(summary.particle_count),
        meanArea: toNumber(summary.mean_area, mean(areas)),
        medianArea: toNumber(summary.median_area, median(areas)),
        stdArea: toNumber(summary.std_area, std(areas)),
        meanDiameter: toNumber(summary.mean_diameter, mean(diameters)),
        medianDiameter: toNumber(summary.median_diameter, median(diameters)),
        stdDiameter: toNumber(summary.std_diameter, std(diameters)),
        areas,
        diameters,
        particles,
        areaUnit: String(summary.area_unit ?? 'px^2'),
        diameterUnit: String(summary.diameter_unit ?? 'px'),
        foregroundPixels: toNumber(summary.foreground_pixels),
        totalPixels: toNumber(summary.total_pixels),
      }
    })

export const aggregateModeRows = (
  rows: ModeImageRow[],
  batchFallback?: Record<string, any> | null,
): ModeBatchSummary => {
  const areas = rows.flatMap((row) => row.areas)
  const diameters = rows.flatMap((row) => row.diameters)
  const volumeFractions = rows.map((row) => row.volumeFraction)
  const particleCounts = rows.map((row) => row.particleCount)
  const foregroundPixels = rows.reduce((sum, row) => sum + row.foregroundPixels, 0)
  const totalPixels = rows.reduce((sum, row) => sum + row.totalPixels, 0)
  const batch = batchFallback ?? {}

  return {
    imageCount: rows.length,
    volumeFraction: toNumber(batch.volume_fraction, totalPixels ? foregroundPixels / totalPixels : mean(volumeFractions)),
    avgVolumeFraction: toNumber(batch.avg_volume_fraction, mean(volumeFractions)),
    avgParticleCount: toNumber(batch.avg_particle_count, mean(particleCounts)),
    particleCountTotal: toNumber(batch.particle_count_total, particleCounts.reduce((sum, count) => sum + count, 0)),
    meanArea: toNumber(batch.mean_area, mean(areas)),
    medianArea: toNumber(batch.median_area, median(areas)),
    stdArea: toNumber(batch.std_area, std(areas)),
    meanDiameter: toNumber(batch.mean_diameter, mean(diameters)),
    medianDiameter: toNumber(batch.median_diameter, median(diameters)),
    stdDiameter: toNumber(batch.std_diameter, std(diameters)),
    areaUnit: String(batch.area_unit ?? rows[0]?.areaUnit ?? 'px^2'),
    diameterUnit: String(batch.diameter_unit ?? rows[0]?.diameterUnit ?? 'px'),
    areas,
    diameters,
    volumeFractions,
    particleCounts,
  }
}

export const buildProbabilityHistogram = (values: number[], desiredBins = 12) => {
  if (!values.length) {
    return {
      labels: [] as string[],
      counts: [] as number[],
      probabilities: [] as number[],
      min: 0,
      max: 0,
    }
  }

  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  if (minValue === maxValue) {
    return {
      labels: [`${minValue.toFixed(2)}`],
      counts: [values.length],
      probabilities: [1],
      min: minValue,
      max: maxValue,
    }
  }

  const bins = Math.min(20, Math.max(6, desiredBins))
  const step = (maxValue - minValue) / bins
  const counts = Array.from({ length: bins }, () => 0)

  values.forEach((value) => {
    const index = value === maxValue ? bins - 1 : Math.min(bins - 1, Math.floor((value - minValue) / step))
    counts[index] += 1
  })

  const labels = counts.map((_, index) => {
    const start = minValue + index * step
    const end = start + step
    return `${start.toFixed(2)} - ${end.toFixed(2)}`
  })

  return {
    labels,
    counts,
    probabilities: counts.map((count) => +(count / values.length).toFixed(4)),
    min: minValue,
    max: maxValue,
  }
}

export const formatMetricValue = (value: number, digits = 2) => {
  if (!Number.isFinite(value)) return '--'
  return value.toFixed(digits)
}

export const formatPercent = (value: number, digits = 2) => `${formatMetricValue(value * 100, digits)}%`
