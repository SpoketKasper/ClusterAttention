try:
  from .run import assignment
except Exception as e:
  print("falling back on assignment:", e)
  from .fallbacks_or_unused.run_fallback import assignment