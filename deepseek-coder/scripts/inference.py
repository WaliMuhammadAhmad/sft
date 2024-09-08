from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your fine-tuned model
model_name = "result"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define your input instruction
input_text = """
Generate a unit test case for the following Java method: JodaTimeSingleFieldPeriodConverter extends AbstractConverter { 
    @Override public Object convert(Object source, TypeToken<?> targetTypeToken) { 
        if (!canHandle(source, targetTypeToken)) { 
            throw new ConverterException(source, targetTypeToken.getRawType()); 
        } 
        Integer period = null; 
        if (source instanceof Number) { 
            period = (Integer) numberToNumberConverter.convert(source, TypeToken.of(Integer.class)); 
        } else if (source instanceof String) { 
            period = (Integer) stringToNumberConverter.convert(source, TypeToken.of(Integer.class)); 
        } 
        Type targetType = targetTypeToken.getType(); 
        if (targetType.equals(Seconds.class)) { 
            return Seconds.seconds(period); 
        } else if (targetType.equals(Minutes.class)) { 
            return Minutes.minutes(period); 
        } else if (targetType.equals(Hours.class)) { 
            return Hours.hours(period); 
        } else if (targetType.equals(Days.class)) { 
            return Days.days(period); 
        } else if (targetType.equals(Weeks.class)) { 
            return Weeks.weeks(period); 
        } else if (targetType.equals(Months.class)) { 
            return Months.months(period); 
        } else if (targetType.equals(Years.class)) { 
            return Years.years(period); 
        } 
        throw new ConverterException(source, targetTypeToken.getRawType()); 
    } 
}
"""

# Encode the input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the output
outputs = model.generate(inputs["input_ids"], max_length=512)

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Unit Test Case:")
print(generated_text)