<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    modelValue: boolean
    imageUrl: string | null
    imageAlt?: string
    title: string
    subtitle?: string
    width?: string
  }>(),
  {
    imageAlt: '图像预览',
    subtitle: '',
    width: 'min(92vw, 1180px)',
  },
)

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()

const bodyRef = ref<HTMLDivElement | null>(null)
const scale = ref(1)
const naturalWidth = ref(0)
const naturalHeight = ref(0)
const fitScale = ref(1)

const visible = computed({
  get: () => props.modelValue,
  set: (value: boolean) => emit('update:modelValue', value),
})

const clampScale = (nextScale: number) => Math.min(4, Math.max(1, Number(nextScale.toFixed(2))))

const clearViewerState = () => {
  scale.value = 1
  naturalWidth.value = 0
  naturalHeight.value = 0
  fitScale.value = 1
}

const resetScale = async () => {
  scale.value = 1
  await nextTick()
  syncFitScale()
}

const zoomIn = () => {
  scale.value = clampScale(scale.value + 0.25)
}

const zoomOut = () => {
  scale.value = clampScale(scale.value - 0.25)
}

const syncFitScale = () => {
  const body = bodyRef.value
  if (!body || !naturalWidth.value || !naturalHeight.value) return
  const computedStyle = window.getComputedStyle(body)
  const horizontalPadding = Number.parseFloat(computedStyle.paddingLeft || '0') + Number.parseFloat(computedStyle.paddingRight || '0')
  const verticalPadding = Number.parseFloat(computedStyle.paddingTop || '0') + Number.parseFloat(computedStyle.paddingBottom || '0')
  const availableWidth = Math.max(body.clientWidth - horizontalPadding, 240)
  const availableHeight = Math.max(body.clientHeight - verticalPadding, 240)
  fitScale.value = Math.min(
    availableWidth / naturalWidth.value,
    availableHeight / naturalHeight.value,
    3,
  )
}

const handleImageLoad = async (event: Event) => {
  const target = event.target as HTMLImageElement | null
  if (!target) return
  naturalWidth.value = target.naturalWidth
  naturalHeight.value = target.naturalHeight
  await nextTick()
  syncFitScale()
}

const handleWheel = (event: WheelEvent) => {
  if (!visible.value) return
  event.preventDefault()
  const delta = event.deltaY < 0 ? 0.15 : -0.15
  scale.value = clampScale(scale.value + delta)
}

const displayStyle = computed(() => {
  if (!naturalWidth.value || !naturalHeight.value) {
    return {
      maxWidth: '100%',
      maxHeight: '100%',
      width: 'auto',
      height: 'auto',
    }
  }

  const nextScale = fitScale.value * scale.value
  return {
    width: `${Math.max(1, Math.round(naturalWidth.value * nextScale))}px`,
    height: `${Math.max(1, Math.round(naturalHeight.value * nextScale))}px`,
    maxWidth: 'none',
    maxHeight: 'none',
  }
})

const displayPercent = computed(() => Math.max(1, Math.round(fitScale.value * scale.value * 100)))

const handleResize = () => {
  if (!visible.value) return
  syncFitScale()
}

watch(
  () => props.modelValue,
  async (nextVisible) => {
    if (!nextVisible) {
      clearViewerState()
      return
    }
    await nextTick()
    syncFitScale()
  },
)

watch(
  () => props.imageUrl,
  () => {
    if (!visible.value) return
    clearViewerState()
  },
)

window.addEventListener('resize', handleResize)

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<template>
  <el-dialog
    v-model="visible"
    :width="width"
    top="4vh"
    destroy-on-close
    append-to-body
    :lock-scroll="false"
    @closed="clearViewerState"
    class="image-preview-dialog"
  >
    <template #header>
      <div class="image-preview-dialog__head">
        <div class="image-preview-dialog__copy">
          <strong>{{ title }}</strong>
          <span v-if="subtitle">{{ subtitle }}</span>
        </div>
        <div class="image-preview-dialog__toolbar">
          <el-button size="small" plain @click="zoomOut">缩小</el-button>
          <span class="image-preview-dialog__scale">{{ displayPercent }}%</span>
          <el-button size="small" plain @click="zoomIn">放大</el-button>
          <el-button size="small" plain @click="resetScale">重置</el-button>
        </div>
      </div>
    </template>
    <div ref="bodyRef" class="image-preview-dialog__body" @wheel.prevent="handleWheel">
      <div class="image-preview-dialog__canvas">
        <img
          v-if="imageUrl"
          :src="imageUrl"
          :alt="imageAlt"
          :style="displayStyle"
          @load="handleImageLoad"
        />
      </div>
    </div>
  </el-dialog>
</template>

<style scoped>
.image-preview-dialog__head {
  display: grid;
  gap: 10px;
}

.image-preview-dialog__copy {
  display: grid;
  gap: 6px;
}

.image-preview-dialog__head strong {
  color: var(--ink);
  font-size: 18px;
}

.image-preview-dialog__head span {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.6;
}

.image-preview-dialog__toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.image-preview-dialog__scale {
  min-width: 48px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  text-align: center;
}

.image-preview-dialog__body {
  min-height: min(72vh, 760px);
  max-height: min(72vh, 760px);
  padding: 24px;
  border-radius: 20px;
  background:
    radial-gradient(circle at top, rgba(57, 84, 112, 0.35), rgba(18, 20, 24, 0.92)),
    rgba(18, 20, 24, 0.94);
  overflow: auto;
}

.image-preview-dialog__canvas {
  width: max-content;
  height: max-content;
  min-width: 100%;
  min-height: calc(min(72vh, 760px) - 48px);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-preview-dialog__body img {
  display: block;
  object-fit: contain;
}

@media (max-width: 760px) {
  .image-preview-dialog__toolbar {
    gap: 10px;
  }

  .image-preview-dialog__body {
    padding: 16px;
  }
}
</style>
