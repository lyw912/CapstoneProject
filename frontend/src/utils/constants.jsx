import {
  ThunderboltOutlined,
  RadarChartOutlined,
  SearchOutlined,
  EditOutlined,
  ControlOutlined
} from '@ant-design/icons';

export const THEME_TOKENS = {
  green: {
    label: 'HKU Green',
    primary: '#115e59',
    primaryDark: '#024638',
    primaryMid: '#2d8b73',
    primarySoft: '#bce6d5',
    trail: 'rgba(17,94,89,.12)',
    accent: '#0b6b58'
  },
  blue: {
    label: 'Blue Tech',
    primary: '#2563eb',
    primaryDark: '#0f2f66',
    primaryMid: '#3b82f6',
    primarySoft: '#bfdbfe',
    trail: 'rgba(37,99,235,.13)',
    accent: '#06b6d4'
  }
};

export const NAV_ITEMS = [
  { key: 'command', label: 'Home', icon: <ThunderboltOutlined /> },
  { key: 'intelligence', label: 'Readout', icon: <RadarChartOutlined /> },
  { key: 'evidence', label: 'Proof', icon: <SearchOutlined /> },
  { key: 'review', label: 'Edit', icon: <EditOutlined /> },
  { key: 'control', label: 'Monitor', icon: <ControlOutlined /> }
];

export const CONFIG_GROUPS = [
  {
    title: 'Foundation Models',
    description: 'Model providers used by analysis, evidence retrieval, and report writing.',
    fields: [
      ['QUERY_ENGINE_API_KEY', 'Evidence model key', 'password'],
      ['QUERY_ENGINE_BASE_URL', 'Evidence model URL', 'text'],
      ['QUERY_ENGINE_MODEL_NAME', 'Evidence model name', 'text'],
      ['MEDIA_ENGINE_API_KEY', 'Media model key', 'password'],
      ['MEDIA_ENGINE_BASE_URL', 'Media model URL', 'text'],
      ['MEDIA_ENGINE_MODEL_NAME', 'Media model name', 'text'],
      ['REPORT_ENGINE_API_KEY', 'Report model key', 'password'],
      ['REPORT_ENGINE_BASE_URL', 'Report model URL', 'text'],
      ['REPORT_ENGINE_MODEL_NAME', 'Report model name', 'text']
    ]
  },
  {
    title: 'Search and Retrieval',
    description: 'External search providers used to collect public evidence.',
    fields: [
      ['SEARCH_TOOL_TYPE', 'Search provider', 'select'],
      ['TAVILY_API_KEY', 'Tavily key', 'password'],
      ['BOCHA_WEB_SEARCH_API_KEY', 'Bocha key', 'password'],
      ['ANSPIRE_API_KEY', 'Anspire key', 'password']
    ]
  },
  {
    title: 'Trace Quality',
    description: 'LangSmith tracing for model calls, timing, errors, and review quality.',
    fields: [
      ['LANGSMITH_TRACING', 'Tracing enabled', 'boolean'],
      ['LANGSMITH_API_KEY', 'LangSmith key', 'password'],
      ['LANGSMITH_ENDPOINT', 'LangSmith endpoint', 'text'],
      ['LANGSMITH_PROJECT', 'LangSmith project', 'text']
    ]
  }
];

export const FLOW_STEPS = [
  { id: 'brief', label: 'Brief', sub: 'Topic', micro: ['Intent', 'Scope', 'Context'] },
  { id: 'collect', label: 'Collect', sub: 'Sources', micro: ['Search', 'Rank', 'Dedup', 'Trust'] },
  { id: 'map', label: 'Map', sub: 'Patterns', micro: ['Stance', 'Sentiment', 'Coverage', 'Divergence'] },
  { id: 'reason', label: 'Reason', sub: 'Tensions', micro: ['Debate', 'Consensus', 'Dissent'] },
  { id: 'verify', label: 'Verify', sub: 'Claims', micro: ['Facts', 'Opinions', 'Bias'] },
  { id: 'write', label: 'Write', sub: 'Report', micro: ['Outline', 'Draft', 'Review', 'Export'] }
];

export const MICRO_STEPS = FLOW_STEPS.flatMap((step, stageIndex) => step.micro.map((name, microIndex) => ({
  id: `${step.id}-${microIndex}`,
  name,
  stageId: step.id,
  stageIndex,
  stageLabel: step.label
})));

export const TRACE_COLORS = {
  chain: '#2563eb',
  llm: '#0891b2',
  tool: '#7c3aed',
  retriever: '#16a34a',
  parser: '#f59e0b',
  unknown: '#64748b',
  'local step': '#2563eb'
};

export const STANCE_COLORS = {
  support: '#024638',
  oppose: '#d9463e',
  neutral: '#4f8f7b',
  official: '#0b6b58',
  background: '#8bb9a8',
  unknown: '#8a978f'
};

export const LAST_QUERY_STORAGE_KEY = 'signal-studio-last-query';
