if __name__ == '__main__':
    '''
    file format:

    0	drugs.com
<Dasabuvir, drugs.com, Viekira Pak>
<Inosine pranobex, drugs.com, isoprinosine-500mg.html>
<Lysergic acid diethylamide, drugs.com, lsd.html>
<Savlon, drugs.com, savlon-antiseptic-cream-leaflet.html>
<Tretinoin, drugs.com, Topical medication>
question template: "Which drug is listed on drugs.com as {head}?"

1	sales
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  2000  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  2001  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  2002  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  2003  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  2004  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  996  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  997  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  998  1>
<Acura CL  Second generation YA4  1, sales, Acura CL  Second generation YA4  1  999  1>
<Acura Legend  Second generation  1, sales, Acura Legend  Second generation  1  986  1>
question template: "What is the sales figure for {head}?"

2	largestCity
<2021 in Mongolia, largestCity, Ulaanbaatar>
<ASEAN    1, largestCity, Jakarta>
<Abbeville County, South Carolina, largestCity, Abbeville, South Carolina>
<Absaroka (proposed state), largestCity, Rapid City, South Dakota>
<Acadia Parish, Louisiana, largestCity, Crowley, Louisiana>
<Accomack County, Virginia, largestCity, Chincoteague, Virginia>
<Ada County, Idaho, largestCity, Boise, Idaho>
<Adair County, Iowa, largestCity, Greenfield, Iowa>
<Adair County, Kentucky, largestCity, Columbia, Kentucky>
<Adair County, Missouri, largestCity, Kirksville, Missouri>
question template: What is the largest city in {head}?

3	fuelType
<12-1 SRW, fuelType, Cetane number>
<12-1 SRW, fuelType, Diesel fuel>
<ABQ RIDE  ABQ RIDE  1, fuelType, Compressed natural gas>
<ABQ RIDE  ABQ RIDE  1, fuelType, Diesel-electric>
<ABQ RIDE  ABQ RIDE  1, fuelType, Gasoline>
<AMC V8 engine, fuelType, Gasoline>
<AMC straight-4 engine, fuelType, Gasoline>
<AMC straight-6 engine, fuelType, Petrol engine>
<Alfa Romeo 12-cylinder engine, fuelType, Gasoline>
<Alfa Romeo 145 and 146  1.4*  1, fuelType, Weber carburetor>
question template: What is the fuel type of {head}?

...
    '''
    import torch

    with open('../relations_clean3.txt', 'r') as f:
        lines = f.readlines()
        llm_question_cache = {}
        idx = 0
        for line in lines:
            if line.startswith('question template:'):
                question = line[len('question template: '):].strip()
                while question[0] == '"':
                    question = question[1:]
                while question[-1] == '"':
                    question = question[:-1]
                llm_question_cache[idx] = question
                idx += 1

        torch.save(llm_question_cache, '../cache/llm_question_cache_final.pt')