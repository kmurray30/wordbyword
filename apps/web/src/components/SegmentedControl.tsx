import "./SegmentedControl.css";

interface SegmentedControlOption<T extends string> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string> {
  options: SegmentedControlOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
}

// A button-based stand-in for <select> - iOS Safari shows its keyboard
// accessory bar's "<" ">" field-navigation arrows whenever the page has
// more than one native form control (input/textarea/select) to hop
// between, so every such control besides the chat textarea itself gets
// replaced with plain buttons like this one.
export function SegmentedControl<T extends string>({ options, value, onChange, disabled }: SegmentedControlProps<T>) {
  return (
    <div className="segmented-control" role="radiogroup">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          role="radio"
          aria-checked={opt.value === value}
          disabled={disabled}
          className={`segmented-control__option${opt.value === value ? " segmented-control__option--active" : ""}`}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
